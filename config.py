# config.py
import yaml
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ExperimentConfig:
    """Конфигурация одного эксперимента"""
    dataset_name: str = "Cora"
    hidden_channels: int = 16
    sparsity_rate: float = 0.0
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    dropout: float = 0.5
    use_mps: bool = False
    train_baseline: bool = True
    retrain_after_prune: bool = True
    runs: int = 3  # Количество запусков для статистики
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

# Конфигурации для разных датасетов
CORR_CONFIG = ExperimentConfig(
    dataset_name="Cora",
    hidden_channels=16,
    epochs=200,
    lr=0.01,
)

CITESEER_CONFIG = ExperimentConfig(
    dataset_name="Citeseer",
    hidden_channels=16,
    epochs=200,
    lr=0.01,
)

PUBMED_CONFIG = ExperimentConfig(
    dataset_name="Pubmed",
    hidden_channels=64,  # Больше из-за размера графа
    epochs=300,
    lr=0.005,
)

# Список уровней разреженности для экспериментов
DEFAULT_SPARSITY_RATES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]