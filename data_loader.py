# data_loader.py
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

def load_dataset(name: str = "Cora", device: str = "cpu"):
    """
    Загружает датасет и перемещает данные на устройство
    
    Args:
        name: Название датасета (Cora, Citeseer, Pubmed)
        device: Устройство (cpu, mps, cuda)
    
    Returns:
        data: Объект с данными
        dataset: Объект датасета
    """
    print(f"Загрузка датасета {name}...")
    
    # Загружаем датасет
    dataset = Planetoid(root=f'./data/{name}', name=name)
    data = dataset[0]
    
    # Перемещаем на устройство
    data = data.to(device)
    
    print(f"  Узлов: {data.num_nodes}")
    print(f"  Ребер: {data.num_edges}")
    print(f"  Признаков: {dataset.num_features}")
    print(f"  Классов: {dataset.num_classes}")
    print(f"  Обучающих узлов: {data.train_mask.sum().item()}")
    print(f"  Валидационных узлов: {data.val_mask.sum().item()}")
    print(f"  Тестовых узлов: {data.test_mask.sum().item()}")
    
    return data, dataset

def get_device(use_mps: bool = True) -> str:
    """
    Определяет доступное устройство
    
    Returns:
        Строка с устройством: 'mps', 'cuda' или 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif use_mps and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'