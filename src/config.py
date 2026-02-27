from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
  task_type: str = 'classification'
  hidden_dim: int = 62
  latent_dim: int = 768
  max_atoms: int = 170
  num_layers: int = 4
  num_attn: int = 4
  batch_size: int = 8
  epoch_size: int = 200
  learning_rate: float = 3e-4
  regularization_scale: float = 4e-4
  beta1: float = 0.9
  beta2: float = 0.98
  num_mc_samples_test: int = 5
  patience: int = 200
  num_train: Optional[int] = None
  optimizer: str = 'Adam' # 'Options : Adam, SGD, RMSProp'
  save_substructures: bool = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TOX_DIR = PROJECT_ROOT / "data" / 'toxicidade'