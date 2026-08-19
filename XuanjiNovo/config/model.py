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
    gradient_clip_algorithm: str = Field("norm", pattern="^(norm|value)$")
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
    log_level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    
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

    def to_flat_dict(self) -> Dict:
        """Convert nested config to flat dictionary for backward compatibility."""
        flat_dict = {
            'random_seed': self.random_seed,
            'log_level': self.log_level,
            # Training fields
            'train_batch_size': self.training.train_batch_size,
            'max_epochs': self.training.max_epochs,
            'warm_up_epochs': self.training.warm_up_epochs,
            'learning_rate': self.training.learning_rate,
            'weight_decay': self.training.weight_decay,
            'gradient_clip_val': self.training.gradient_clip_val,
            'gradient_clip_algorithm': self.training.gradient_clip_algorithm,
            'accumulate_grad_batches': self.training.accumulate_grad_batches,
            'sync_batchnorm': self.training.sync_batchnorm,
            'SWA': self.training.SWA,
            # Model fields
            'dim_model': self.model.dim_model,
            'n_head': self.model.n_head,
            'dim_feedforward': self.model.dim_feedforward,
            'n_layers': self.model.n_layers,
            'dropout': self.model.dropout,
            'max_length': self.model.max_length,
            # Decoding fields
            'PMC_enable': self.decoding.PMC_enable,
            'mass_control_tol': self.decoding.mass_control_tol,
            'n_beams': self.decoding.n_beams,
            'refine_iters': self.decoding.refine_iters,
            # Data fields
            'n_peaks': self.data.n_peaks,
            'min_mz': self.data.min_mz,
            'max_mz': self.data.max_mz,
            'min_intensity': self.data.min_intensity,
            'remove_precursor_tol': self.data.remove_precursor_tol,
            'max_charge': self.data.max_charge,
            'precursor_mass_tol': self.data.precursor_mass_tol,
            'isotope_error_range': self.data.isotope_error_range,
            # Mask schedule
            'mask_schedule': {
                'initial_peek': self.mask_schedule.initial_peek,
                'epoch_decay': self.mask_schedule.epoch_decay,
                'min_peek': self.mask_schedule.min_peek,
            },
            # Residues
            'residues': self.residues,
        }
        
        # Add optional fields if they exist
        if self.neptune_config:
            flat_dict['neptune_config'] = self.neptune_config
        if self.custom_metadata:
            flat_dict['custom_metadata'] = self.custom_metadata
            
        return flat_dict

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'XuanjiNovoConfig':
        """Create configuration from YAML file."""
        import yaml
        with open(yaml_path) as f:
            flat_config = yaml.safe_load(f)
        
        # Transform flat config to nested structure
        nested_dict = {
            'random_seed': flat_config.get('random_seed', -1),
            'log_level': flat_config.get('log_level', 'INFO'),
            'training': {
                'train_batch_size': flat_config.get('train_batch_size', 2800),
                'max_epochs': flat_config.get('max_epochs', 150),
                'warm_up_epochs': flat_config.get('warm_up_epochs', 1),
                'learning_rate': flat_config.get('learning_rate', 0.00035),
                'weight_decay': flat_config.get('weight_decay', 8e-5),
                'gradient_clip_val': flat_config.get('gradient_clip_val', 2.5),
                'gradient_clip_algorithm': flat_config.get('gradient_clip_algorithm', 'norm'),
                'accumulate_grad_batches': flat_config.get('accumulate_grad_batches', 1),
                'sync_batchnorm': flat_config.get('sync_batchnorm', False),
                'SWA': flat_config.get('SWA', False),
            },
            'model': {
                'dim_model': flat_config.get('dim_model', 400),
                'n_head': flat_config.get('n_head', 8),
                'dim_feedforward': flat_config.get('dim_feedforward', 1024),
                'n_layers': flat_config.get('n_layers', 9),
                'dropout': flat_config.get('dropout', 0.18),
                'max_length': flat_config.get('max_length', 40),
            },
            'decoding': {
                'PMC_enable': flat_config.get('PMC_enable', True),
                'mass_control_tol': flat_config.get('mass_control_tol', 0.1),
                'n_beams': flat_config.get('n_beams', 5),
                'refine_iters': flat_config.get('refine_iters', 3),
            },
            'data': {
                'n_peaks': flat_config.get('n_peaks', 800),
                'min_mz': flat_config.get('min_mz', 1.0),
                'max_mz': flat_config.get('max_mz', 6500.0),
                'min_intensity': flat_config.get('min_intensity', 0.0),
                'remove_precursor_tol': flat_config.get('remove_precursor_tol', 1.0),
                'max_charge': flat_config.get('max_charge', 10),
                'precursor_mass_tol': flat_config.get('precursor_mass_tol', 50.0),
                'isotope_error_range': tuple(flat_config.get('isotope_error_range', [0, 1])),
            },
            'mask_schedule': flat_config.get('mask_schedule', {
                'initial_peek': 0.93,
                'epoch_decay': 0.01,
                'min_peek': 0.00,
            }),
            'residues': flat_config.get('residues', {}),
            'neptune_config': flat_config.get('neptune_config'),
            'custom_metadata': flat_config.get('custom_metadata'),
        }
        
        return cls(**nested_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'XuanjiNovoConfig':
        """Create configuration from dictionary."""
        # Check if it's already nested structure by looking for 'training' key
        if 'training' in config_dict and isinstance(config_dict.get('training'), dict):
            # Already nested, use directly
            return cls(**config_dict)
        else:
            # Flat structure, transform to nested
            nested_dict = {
                'random_seed': config_dict.get('random_seed', -1),
                'log_level': config_dict.get('log_level', 'INFO'),
                'training': {
                    'train_batch_size': config_dict.get('train_batch_size', 2800),
                    'max_epochs': config_dict.get('max_epochs', 150),
                    'warm_up_epochs': config_dict.get('warm_up_epochs', 1),
                    'learning_rate': config_dict.get('learning_rate', 0.00035),
                    'weight_decay': config_dict.get('weight_decay', 8e-5),
                    'gradient_clip_val': config_dict.get('gradient_clip_val', 2.5),
                    'gradient_clip_algorithm': config_dict.get('gradient_clip_algorithm', 'norm'),
                    'accumulate_grad_batches': config_dict.get('accumulate_grad_batches', 1),
                    'sync_batchnorm': config_dict.get('sync_batchnorm', False),
                    'SWA': config_dict.get('SWA', False),
                },
                'model': {
                    'dim_model': config_dict.get('dim_model', 400),
                    'n_head': config_dict.get('n_head', 8),
                    'dim_feedforward': config_dict.get('dim_feedforward', 1024),
                    'n_layers': config_dict.get('n_layers', 9),
                    'dropout': config_dict.get('dropout', 0.18),
                    'max_length': config_dict.get('max_length', 40),
                },
                'decoding': {
                    'PMC_enable': config_dict.get('PMC_enable', True),
                    'mass_control_tol': config_dict.get('mass_control_tol', 0.1),
                    'n_beams': config_dict.get('n_beams', 5),
                    'refine_iters': config_dict.get('refine_iters', 3),
                },
                'data': {
                    'n_peaks': config_dict.get('n_peaks', 800),
                    'min_mz': config_dict.get('min_mz', 1.0),
                    'max_mz': config_dict.get('max_mz', 6500.0),
                    'min_intensity': config_dict.get('min_intensity', 0.0),
                    'remove_precursor_tol': config_dict.get('remove_precursor_tol', 1.0),
                    'max_charge': config_dict.get('max_charge', 10),
                    'precursor_mass_tol': config_dict.get('precursor_mass_tol', 50.0),
                    'isotope_error_range': tuple(config_dict.get('isotope_error_range', [0, 1])),
                },
                'mask_schedule': config_dict.get('mask_schedule', {
                    'initial_peek': 0.93,
                    'epoch_decay': 0.01,
                    'min_peek': 0.00,
                }),
                'residues': config_dict.get('residues', {}),
                'neptune_config': config_dict.get('neptune_config'),
                'custom_metadata': config_dict.get('custom_metadata'),
            }
            return cls(**nested_dict)
