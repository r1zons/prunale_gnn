# config.py (обновлённый)
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
    hidden_channels=64,
    epochs=300,
    lr=0.005,
)

# НОВЫЕ КОНФИГУРАЦИИ
TEXAS_CONFIG = ExperimentConfig(
    dataset_name="Texas",
    hidden_channels=16,
    epochs=200,
    lr=0.01,
    # WebKB датасеты обычно меньше, можно уменьшить hidden_channels
)

ACTOR_CONFIG = ExperimentConfig(
    dataset_name="Actor",
    hidden_channels=64,  # Actor имеет больше признаков (931), чем другие датасеты
    epochs=200,
    lr=0.01,
    # Actor имеет 5 классов и 931 признак, может потребоваться больший скрытый слой
)



# Карта конфигураций для удобного доступа
CONFIG_MAP = {
    "Cora": CORR_CONFIG,
    "Citeseer": CITESEER_CONFIG,
    "Pubmed": PUBMED_CONFIG,
    "Texas": TEXAS_CONFIG,
    "Actor": ACTOR_CONFIG,
}

# Список уровней разреженности для экспериментов
DEFAULT_SPARSITY_RATES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]


def get_config_for_dataset(dataset_name: str) -> ExperimentConfig:
    """Возвращает конфигурацию для указанного датасета"""
    if dataset_name in CONFIG_MAP:
        return CONFIG_MAP[dataset_name]
    else:
        # Конфигурация по умолчанию для новых датасетов
        return ExperimentConfig(dataset_name=dataset_name)