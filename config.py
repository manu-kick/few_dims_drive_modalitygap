# config.py
from dataclasses import dataclass, asdict
from typing import Any, Dict

@dataclass
class Config:
    dataset_name: str = "flickr30k"
    dataset_root: str = "/mnt/media/eleonora/flickr30k"

    run_type: str = "train"  # "train" / "test" 
    embedding_dim: int = 128
    output_dim: int = 128
    reproject_with_shared_head: bool = False
    w2v_path: str = "./GoogleNews-vectors-negative300.bin"

    batch_size: int = 32
    num_iterations: int = 10000
    device: str = "cuda"
    seed: int = 42
    eval_every: int = 200

    loss_type: str = "anchor"  # anchor / volume / centroids / area
    lr: float = 1e-4
    normalization: bool = True

    contra_temp_init: float = 0.07
    contra_temp_learnable: bool = True

    plot_path: str = "./plot"
    wandb: bool = True
    run: str = ""  # filled after overrides
    wandb_artifact_name: str = ""  # e.g. "rucci-emanuele-personal/flickr30k_ContrastiveLearning/best-model:v160"

    def finalize(self):
        if not self.run:
            self.run = f"SharedHead_tau_{self.contra_temp_init}_learnable({self.contra_temp_learnable})_embdim{self.embedding_dim}"

    def log_config(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        cf = cls(**d)
        cf.finalize()
        return cf

