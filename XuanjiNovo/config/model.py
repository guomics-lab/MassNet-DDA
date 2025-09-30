"""Configuration validation models for XuanjiNovo."""

from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, validator
import torch

class MaskSchedule(BaseModel):
    """Mask schedule configuration."""
    initial_peek: float = Field(0.93, ge=0.0, le=1.0)
    epoch_decay: float = Field(0.01, ge=0.0)
    min_peek: float = Field(0.00, ge=0.0, le=1.0)

    @validator('min_peek')
    def min_peek_less_than_initial(cls, v, values):
        if 'initial_peek' in values and v > values['initial_peek']:
            raise ValueError('min_peek must be less than or equal to initial_peek')
        return v

class TrainingConfig(BaseModel):
    """Training-specific configuration."""
    train_batch_size: int = Field(2800, gt=0)
    max_epochs: int = Field(150, gt=0)
    warm_up_epochs: int = Field(1, ge=0)
    learning_rate: float = Field(0.00035, gt=0)
    weight_decay: float = Field(8e-5, ge=0)
    gradient_clip_val: float = Field(2.5, gt=0)
    gradient_clip_algorithm: str = Field("norm", regex="^(norm|value)$")
    accumulate_grad_batches: int = Field(1, gt=0)
    sync_batchnorm: bool = False
    SWA: bool = False  # StochasticWeightAveraging

class ModelConfig(BaseModel):
    """Model architecture configuration."""
    dim_model: int = Field(400, gt=0)
    n_head: int = Field(8, gt=0)
    dim_feedforward: int = Field(1024, gt=0)
    n_layers: int = Field(9, gt=0)
    dropout: float = Field(0.18, ge=0.0, le=1.0)
    max_length: int = Field(40, gt=0)

class DecodingConfig(BaseModel):
    """Decoding configuration."""
    PMC_enable: bool = True
    mass_control_tol: float = Field(0.1, gt=0)
    n_beams: int = Field(5, gt=0)
    refine_iters: int = Field(3, gt=0)

class DataConfig(BaseModel):
    """Data processing configuration."""
    n_peaks: int = Field(800, gt=0)
    min_mz: float = Field(1.0, ge=0)
    max_mz: float = Field(6500.0, gt=0)
    min_intensity: float = Field(0.0, ge=0)
    remove_precursor_tol: float = Field(1.0, gt=0)
    max_charge: int = Field(10, gt=0)
    precursor_mass_tol: float = Field(50.0, gt=0)
    isotope_error_range: Tuple[int, int] = Field((0, 1))

    @validator('isotope_error_range')
    def validate_isotope_range(cls, v):
        if v[0] > v[1]:
            raise ValueError('isotope_error_range[0] must be <= isotope_error_range[1]')
        return v

class XuanjiNovoConfig(BaseModel):
    """Complete configuration for XuanjiNovo."""
    # Basic settings
    random_seed: int = -1
    log_level: str = Field("INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    
    # Component configurations
    training: TrainingConfig
    model: ModelConfig
    decoding: DecodingConfig
    data: DataConfig
    mask_schedule: MaskSchedule

    # Residues configuration
    residues: Dict[str, float]

    # Optional configurations
    neptune_config: Optional[Dict] = None
    custom_metadata: Optional[Dict] = None

    @validator('model')
    def validate_model_heads(cls, v):
        if v.dim_model % v.n_head != 0:
            raise ValueError('dim_model must be divisible by n_head')
        return v

    def get_device_config(self) -> Dict:
        """Get device-specific configuration."""
        return {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] 
                        if torch.cuda.is_available() else []
        }

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to a file."""
        with open(filepath, 'w') as f:
            f.write(self.json(indent=2))

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'XuanjiNovoConfig':
        """Create configuration from YAML file."""
        import yaml
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'XuanjiNovoConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
