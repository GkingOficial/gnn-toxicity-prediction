from dataclasses import dataclass

@dataclass
class TrainConfig:
  dim1: int = 62
  dim2: int = 768
  max_atoms: int = 170
  num_layer: int = 4
  batch_size: int = 8
  epoch_size: int = 200
  learning_rate: float = 3e-4
  regularization_scale: float = 4e-4
  beta1: float = 0.9
  beta2: float = 0.98
  num_mc_samples_test: int = 5
  patience: int = 200