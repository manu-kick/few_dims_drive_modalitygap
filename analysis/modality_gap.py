# This files implements several computation of the modality gap as described in several papers

# Modality gap as in the original paper
import torch
import wandb


def L2M(cf,metric,text_embeddings,vision_embeddings,iterations):
    # Compute the mean for each modality
    # Compute the L2 distance between the means taken pairwise
    # Also compute the overall average between the three pairs
    text_embeddings = torch.Tensor(text_embeddings)
    vision_embeddings = torch.Tensor(vision_embeddings)
    vision_embeddings = torch.Tensor(vision_embeddings)

    mean_text = text_embeddings.mean(dim=0)
    mean_vision = vision_embeddings.mean(dim=0)

    l2_text_vision = torch.norm(mean_text - mean_vision, p=2).item()



    if cf.wandb:
        wandb.log({f'{metric}_gap': {
            'text_vision': l2_text_vision,
        }})

    return {
        'text_vision': l2_text_vision,
    }


# Relative modality gap as in https://openreview.net/pdf?id=uAFHCZRmXk
def rmg_numerator(mod1, mod2):
    # compute the average cosine dissimilarity (1-cos(x,y))/2 between matching pairs
    return torch.mean(1 - torch.nn.functional.cosine_similarity(mod1, mod2)).item()

def rmg_denominator(mod1, mod2 , numerator):
    N = mod1.shape[0]
    factor_multiplier = 1/((2*N)*(N-1))
    # ---- Intra-modality dissimilarities ----
    # Cosine similarity matrices
    sim_mod1 = torch.nn.functional.cosine_similarity(mod1.unsqueeze(1), mod1.unsqueeze(0), dim=-1)  # [N, N]
    sim_mod2 = torch.nn.functional.cosine_similarity(mod2.unsqueeze(1), mod2.unsqueeze(0), dim=-1)  # [N, N]

    # Cosine dissimilarity matrices (1 - similarity) / 2
    dissim_mod1 = (1 - sim_mod1) / 2  # [N, N]
    dissim_mod2 = (1 - sim_mod2) / 2  # [N, N]

    # We only want the upper triangle (excluding diagonal) for intra-modality pairs
    intra_mod1 = dissim_mod1.triu(diagonal=1).sum().item()
    intra_mod2 = dissim_mod2.triu(diagonal=1).sum().item()

    return (factor_multiplier * (intra_mod1 + intra_mod2))+ numerator

def RMG(cf,metric,text_embeddings,vision_embeddings,iterations):
    couple_modalities = [('text','vision')]

    embeddings = {
        'text':  torch.Tensor(text_embeddings),
        'vision': torch.Tensor(vision_embeddings)
    }
    rmg = dict()
    for couple in couple_modalities:
        mod1, mod2 = couple
        numerator = rmg_numerator(embeddings[mod1], embeddings[mod2])
        denominator = rmg_denominator(embeddings[mod1], embeddings[mod2], numerator)
        rmg_value = numerator / denominator
        rmg[f'{mod1}_{mod2}'] = rmg_value


        if cf.wandb:
            wandb.log({f'{metric}_gap': {
                f'{mod1}_{mod2}': rmg_value,
            }})

    return rmg
    # Not needed in this case
    # overall_rmg = sum(rmg.values()) / len(rmg)
    # rmg['overall'] = overall_rmg
    # if cf.wandb:
    #     wandb.log({f'{metric}_gap': {
    #         'mean': overall_rmg
    #     }})

# L2 Instance gap
def L2I(cf,metric,text_embeddings,vision_embeddings,iterations):
    # average l2 norm between matching pairs
    text_embeddings = torch.Tensor(text_embeddings)
    vision_embeddings = torch.Tensor(vision_embeddings)

    # count_samples = text_embeddings.shape[0]
    l2i_text_vision = torch.norm(text_embeddings - vision_embeddings, p=2, dim=-1).mean().item()
    
    
    if cf.wandb:
        wandb.log({f'{metric}_gap': {
            'text_vision': l2i_text_vision,
        }})

    return {
        'text_vision': l2i_text_vision,
    }

def compute_gap(cf,metric,text_embeddings,vision_embeddings,iterations):
    if metric=='L2M':
        return L2M(cf,metric,text_embeddings,vision_embeddings,iterations)
    elif metric=='RMG':
        return RMG(cf,metric,text_embeddings,vision_embeddings,iterations)
    elif metric=='L2I':
        return L2I(cf,metric,text_embeddings,vision_embeddings,iterations)
    else:
        raise ValueError(f'Unknown metric {metric}')
