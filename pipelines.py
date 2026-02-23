# pipelines.py
from analysis.viz import visualize_3d
from metrics.loss import compute_loss_anchor
from metrics.loss import *
import torch
import torch.nn.functional as F
import numpy as np
import wandb
import os
from tqdm import tqdm

from analysis.gap_mean_differences import gap_mean_differences
from analysis.gap_embedding_dim_pairs import gap_embedding_dim_pairs
from analysis.fisher_cumulative_expl_var import fisher_and_cumulative_explained_variance
from analysis.intrinsic_dimensions import intrinsic_dimension_mle
from analysis.modality_gap import compute_gap

from utils import log_model_to_wandb, save_checkpoint
from metrics.retrieval import retrieval

def get_loss(loss_type, text_embedding, vision_embedding, bs, contra_temp):
    if loss_type == 'anchor':
        loss = compute_loss_anchor(text_embedding, vision_embedding, bs, contra_temp)
    elif loss_type == 'centroids':
        print("loss not implemented")
        return 0
    elif loss_type == 'volume':
        print("loss not implemented")
        return 0
    elif loss_type == 'area':
        print("loss not implemented")
        return 0
    elif loss_type == 'anchor_align_unif':
        print("loss not implemented")
        return 0
    else:
        print("loss not implemented")
        return 0
    return loss

def get_metric(metric_name, text_embedding, vision_embedding, bs):
    if metric_name == 'retrieval_at_1':
        metric = retrieval(text_embedding, vision_embedding, top_k=1)
    elif metric_name == 'retrieval_at_5':
        metric = retrieval(text_embedding, vision_embedding, top_k=5)
    elif metric_name == 'retrieval_at_10':
        metric = retrieval(text_embedding, vision_embedding, top_k=10)
    else:
        print("metric not implemented")
        return 0
    return metric

def eval(cf, test_loader, text_encoder, vision_encoder, shared_head, device, iteration, contra_temp):
    text_encoder.eval()
    vision_encoder.eval()
    if shared_head is not None:
        shared_head.eval()

    text_embeddings = []
    vision_embeddings = []

    # NEW: store retrieval@1 per batch
    retrieval_at_1_batches = []

    with torch.no_grad():
        for batch in test_loader:
            images, captions, fns, cap_idxs = batch
            images = images.to(device)

            text_emb = text_encoder(captions)
            vision_emb = vision_encoder(images)

            if shared_head is not None:
                text_emb = shared_head(text_emb)
                vision_emb = shared_head(vision_emb)

            text_emb = F.normalize(text_emb, dim=-1)
            vision_emb = F.normalize(vision_emb, dim=-1)

            # NEW: retrieval@1 on THIS batch (torch)
            r1 = get_metric("retrieval_at_1", text_emb, vision_emb, text_emb.size(0))
            retrieval_at_1_batches.append(float(r1))

            # keep embeddings for global analyses (numpy)
            text_embeddings.append(text_emb.detach().cpu().numpy())
            vision_embeddings.append(vision_emb.detach().cpu().numpy())

    # numpy arrays for analysis
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    vision_embeddings = np.concatenate(vision_embeddings, axis=0)

    # analysis / visualizations and gaps (unchanged)
    visualize_3d(cf, text_embeddings, vision_embeddings, iteration)
    gap_mean_differences(cf, text_embeddings, vision_embeddings, iteration)
    gap_embedding_dim_pairs(cf, text_embeddings, vision_embeddings, iteration)
    fisher_and_cumulative_explained_variance(cf, text_embeddings, vision_embeddings, iteration)
    intrinsic_dimension_mle(cf, text_embeddings, vision_embeddings, iteration)

    gaps = {}
    for i in ['L2M', 'RMG', 'L2I']:
        gaps[i] = compute_gap(cf, i, text_embeddings, vision_embeddings, iteration)

    # NEW: mean of per-batch retrieval@1
    metrics_results = {}
    metrics_results["retrieval_at_1"] = float(np.mean(retrieval_at_1_batches)) if len(retrieval_at_1_batches) > 0 else 0.0

    # val loss (same as before)
    text_t = torch.from_numpy(text_embeddings).to(device=device, dtype=torch.float32)
    vis_t  = torch.from_numpy(vision_embeddings).to(device=device, dtype=torch.float32)

    temp_for_eval = contra_temp if cf.contra_temp_learnable else cf.contra_temp_init
    eval_loss = get_loss(cf.loss_type, text_t, vis_t, text_t.size(0), temp_for_eval)

    print(
        f"Iteration {iteration} ==> Validation Loss = {eval_loss.item():.4f} "
        f"| L2M Gap: {gaps['L2M']['text_vision']:.4f} "
        f"| RMG Gap: {gaps['RMG']['text_vision']:.4f} "
        f"| L2I Gap: {gaps['L2I']['text_vision']:.4f} "
        f"| retrieval@1 : {metrics_results['retrieval_at_1']:.4f}"
    )

    if cf.wandb:
        wandb.log({
            "validation_loss": eval_loss.item(),
            "L2M_gap": gaps['L2M']['text_vision'],
            "RMG_gap": gaps['RMG']['text_vision'],
            "L2I_gap": gaps['L2I']['text_vision'],
            "retrieval_at_1": metrics_results["retrieval_at_1"],
            "iteration": iteration
        })

    return eval_loss.item(), gaps

def train_model_with_visualization(
    cf,
    text_encoder,
    vision_encoder,
    shared_head,
    train_loader,
    test_loader,
    optimizer,
    device,
    num_iterations,
    contra_temp,
    save_local: bool = False,
    save_dir: str = "checkpoints",
    save_name: str = "best.pt",
    wandb_artifact_name: str = "best-model",
):
    """
    - salva il best su W&B Artifacts quando migliora val_loss
    - (opzionale) salva anche localmente se save_local=True
    """

    if not cf.contra_temp_learnable:
        contra_temp.requires_grad = False

    text_encoder.train()
    vision_encoder.train()
    if shared_head is not None:
        shared_head.train()

    iteration = 0
    best_val_loss = float("inf")

    # eval iniziale
    with torch.no_grad():
        val_loss, gaps = eval(
            cf, test_loader, text_encoder, vision_encoder, shared_head,
            device, iteration, contra_temp
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        # W&B artifact (versionato)
        if cf.wandb:
            log_model_to_wandb(
                text_encoder=text_encoder,
                vision_encoder=vision_encoder,
                shared_head=shared_head,
                optimizer=optimizer,
                contra_temp=contra_temp,
                iteration=iteration,
                best_val_loss=best_val_loss,
                cf=cf,
                artifact_name=wandb_artifact_name,
                artifact_alias="latest",
                extra_metadata={
                    "L2M_gap": float(gaps["L2M"]["text_vision"]),
                    "RMG_gap": float(gaps["RMG"]["text_vision"]),
                    "L2I_gap": float(gaps["L2I"]["text_vision"]),
                }
            )
            wandb.log({"best_val_loss": best_val_loss})

        # opzionale: local checkpoint
        if save_local:
            path = save_checkpoint(
                save_dir=save_dir,
                filename=save_name,
                text_encoder=text_encoder,
                vision_encoder=vision_encoder,
                shared_head=shared_head,
                optimizer=optimizer,
                contra_temp=contra_temp,
                iteration=iteration,
                best_val_loss=best_val_loss,
                cf=cf,
            )
            print(f"[CKPT] Saved local best @ {path} (val_loss={best_val_loss:.6f})")

    # torna in train
    text_encoder.train()
    vision_encoder.train()
    if shared_head is not None:
        shared_head.train()

    running_loss = 0.0
    tq = tqdm(range(num_iterations), total=num_iterations, desc="Training")
    train_iterator = iter(train_loader)

    for _ in tq:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        images, captions, fns, cap_idxs = batch
        images = images.to(device)

        # forward
        text_emb = text_encoder(captions)
        vision_emb = vision_encoder(images)

        if shared_head is not None:
            text_emb = shared_head(text_emb)
            vision_emb = shared_head(vision_emb)

        if cf.normalization:
            text_emb = F.normalize(text_emb, dim=-1)
            vision_emb = F.normalize(vision_emb, dim=-1)

        loss = get_loss(cf.loss_type, text_emb, vision_emb, text_emb.size(0), contra_temp)

        if cf.wandb:
            wandb.log({"train_loss": loss.item(), "iteration": iteration})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1
        running_loss += loss.item()
        tq.set_postfix({"loss": loss.item(), "best_val": best_val_loss})

        # eval periodica
        if iteration % cf.eval_every == 0:
            with torch.no_grad():
                val_loss, gaps = eval(
                    cf, test_loader, text_encoder, vision_encoder, shared_head,
                    device, iteration, contra_temp
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"[BEST] iter={iteration} val_loss={best_val_loss:.6f}")

                # W&B artifact
                if cf.wandb:
                    log_model_to_wandb(
                        text_encoder=text_encoder,
                        vision_encoder=vision_encoder,
                        shared_head=shared_head,
                        optimizer=optimizer,
                        contra_temp=contra_temp,
                        iteration=iteration,
                        best_val_loss=best_val_loss,
                        cf=cf,
                        artifact_name=wandb_artifact_name,
                        artifact_alias="latest",
                        extra_metadata={
                            "L2M_gap": float(gaps["L2M"]["text_vision"]),
                            "RMG_gap": float(gaps["RMG"]["text_vision"]),
                            "L2I_gap": float(gaps["L2I"]["text_vision"]),
                        }
                    )
                    wandb.log({"best_val_loss": best_val_loss})

            # torna in train
            text_encoder.train()
            vision_encoder.train()
            if shared_head is not None:
                shared_head.train()

            if cf.wandb and cf.contra_temp_learnable:
                wandb.log({"contra_temp": float(contra_temp.item()), "iteration": iteration})

    epoch_loss = running_loss / max(1, iteration)
    print(f"Training completed. Average Loss: {epoch_loss:.4f}")
    if cf.wandb:
        wandb.log({"mean_train_loss": epoch_loss})

    return best_val_loss

def test_model_against_tasks(cf, text_encoder, vision_encoder, shared_head, test_loader, device):
    # ---- load best checkpoint from artifact (come prima) ----
    api = wandb.Api(timeout=120)
    artifact = api.artifact('rucci-emanuele-personal/flickr30k_ContrastiveLearning/best-model:v160')
    artifact_dir = artifact.download()
    best_model_path = os.path.join(artifact_dir, "model.pt")

    checkpoint = torch.load(best_model_path, map_location=device)

    text_encoder.load_state_dict(checkpoint["text_encoder_state"])
    vision_encoder.load_state_dict(checkpoint["vision_encoder_state"])
    if cf.reproject_with_shared_head and checkpoint.get("shared_head_state") is not None:
        shared_head.load_state_dict(checkpoint["shared_head_state"])

    print(
        f"Best checkpoint loaded from W&B artifact '{cf.wandb_artifact_name}' "
        f"(iteration {checkpoint['iteration']}, val_loss {checkpoint['best_val_loss']})"
    )

    # ---- eval on test set ----
    metrics_results = {"retrieval_at_1": [], "retrieval_at_5": [], "retrieval_at_10": []}

    text_encoder.eval()
    vision_encoder.eval()
    if shared_head is not None:
        shared_head.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images, captions, fns, cap_idxs = batch
            images = images.to(device)

            text_emb = text_encoder(captions)
            vision_emb = vision_encoder(images)

            if shared_head is not None:
                text_emb = shared_head(text_emb)
                vision_emb = shared_head(vision_emb)

            text_emb = F.normalize(text_emb, dim=-1)
            vision_emb = F.normalize(vision_emb, dim=-1)
            
            for metric_name in metrics_results.keys():
                metrics_results[metric_name].append(
                    get_metric(metric_name, text_emb, vision_emb, text_emb.size(0))
                )
    # ---- aggregate ----
    for metric_name, values in metrics_results.items():
        mean_value = np.mean(values)
        metrics_results[metric_name] = mean_value
        print(f"{metric_name}: {mean_value:.4f}")


    # ---- log to the *existing* run, WITHOUT creating new runs ----
    if cf.wandb:
        # Case A: there's already an active run (best case)
        if wandb.run is not None:
                for metric_name, mean_value in metrics_results.items():
                    wandb.log({f"test_{metric_name}": mean_value})  
        else:
            # Case B: attach/resume an existing run by ID
            # Prefer: take run_id from checkpoint OR cf (you should store this during training)
            run_id = None
            if isinstance(checkpoint, dict):
                run_id = checkpoint.get("wandb_run_id", None)
            if run_id is None:
                run_id = getattr(cf, "wandb_run_id", None)

            if run_id is None:
                # Create a new run if we can't find an existing one (last resort)
                print("Warning: No existing W&B run found. Creating a new run for test results.")
                run = wandb.init(
                    project=f"{cf.dataset_name}_ContrastiveLearning",
                    name=f"{cf.run}_test",
                    config=cf.log_config(),
                    reinit=True,
                    settings=wandb.Settings(init_timeout=120),
                )
                for metric_name, mean_value in metrics_results.items():
                    wandb.log({f"test_{metric_name}": mean_value})
                run.finish()