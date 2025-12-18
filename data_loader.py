# data_loader.py (исправленная версия)
import torch
from torch_geometric.datasets import Planetoid, WebKB, Actor
import torch_geometric.transforms as T
import numpy as np

def load_dataset(name: str = "Cora", device: str = "cpu"):
    """
    Загружает датасет и перемещает данные на устройство
    
    Args:
        name: Название датасета (Cora, Citeseer, Pubmed, Texas, Actor)
        device: Устройство (cpu, mps, cuda)
    
    Returns:
        data: Объект с данными
        dataset: Объект датасета
    """
    print(f"Загрузка датасета {name}...")
    
    # Создаём директорию для данных
    import os
    os.makedirs(f'./data/{name}', exist_ok=True)
    
    # Загружаем соответствующий датасет
    if name in ["Cora", "Citeseer", "Pubmed"]:
        # Стандартные датасеты Planetoid
        dataset = Planetoid(root=f'./data/{name}', name=name)
        data = dataset[0]
        
    elif name in ["Texas", "Cornell", "Wisconsin"]:
        # Датсеты из WebKB (веб-страницы университетов)
        dataset = WebKB(root=f'./data/{name}', name=name)
        data = dataset[0]
        
        # WebKB датасеты имеют другую структуру масок
        # train_mask, val_mask, test_mask - это тензоры формы [num_nodes, num_classes]
        # Нужно преобразовать в одномерные булевы маски
        data = fix_webkb_masks(data)
        
    elif name == "Actor":
        # Датсет актёров из социальной сети
        dataset = Actor(root=f'./data/{name}')
        data = dataset[0]
        
        # Actor датасет также имеет проблемы с масками
        # В нём есть только train_mask и test_mask, нет val_mask
        data = fix_actor_masks(data)
        
    else:
        raise ValueError(f"Неизвестный датасет: {name}")
    
    # Проверяем наличие масок для train/val/test
    # Если масок нет, создаём случайное разбиение
    if not hasattr(data, 'train_mask') or data.train_mask is None:
        print(f"  ⚠️ В датасете {name} нет предопределённых масок. Создаю случайное разбиение...")
        data = create_random_split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    # Проверяем и фиксим размерности масок
    data = fix_mask_dimensions(data)
    
    # Перемещаем на устройство
    data = data.to(device)
    
    print(f"  Узлов: {data.num_nodes}")
    print(f"  Ребер: {data.num_edges}")
    print(f"  Признаков: {dataset.num_features}")
    print(f"  Классов: {dataset.num_classes}")
    
    if hasattr(data, 'train_mask'):
        mask_shape = data.train_mask.shape
        mask_sum = data.train_mask.sum().item() if hasattr(data.train_mask, 'sum') else 'N/A'
        print(f"  Обучающих узлов: {mask_sum}, форма маски: {mask_shape}")
    if hasattr(data, 'val_mask'):
        mask_shape = data.val_mask.shape
        mask_sum = data.val_mask.sum().item() if hasattr(data.val_mask, 'sum') else 'N/A'
        print(f"  Валидационных узлов: {mask_sum}, форма маски: {mask_shape}")
    if hasattr(data, 'test_mask'):
        mask_shape = data.test_mask.shape
        mask_sum = data.test_mask.sum().item() if hasattr(data.test_mask, 'sum') else 'N/A'
        print(f"  Тестовых узлов: {mask_sum}, форма маски: {mask_shape}")
    
    return data, dataset

def fix_webkb_masks(data):
    """
    Исправляет маски для WebKB датасетов (Texas, Cornell, Wisconsin)
    
    В WebKB датасетах маски имеют форму [num_nodes, num_classes]
    Нужно преобразовать в одномерные булевы маски
    """
    print(f"  Исправление масок для WebKB датасета...")
    
    # Сохраняем оригинальные маски для отладки
    if hasattr(data, 'train_mask'):
        original_shape = data.train_mask.shape
        print(f"    Оригинальная форма train_mask: {original_shape}")
        
        # Если маска двумерная, преобразуем в одномерную
        if len(original_shape) == 2:
            # Берем первый ненулевой элемент в каждой строке
            # В WebKB маска обычно one-hot по классам
            data.train_mask = data.train_mask.any(dim=1)
            print(f"    Исправленная форма train_mask: {data.train_mask.shape}")
    
    if hasattr(data, 'val_mask'):
        original_shape = data.val_mask.shape
        print(f"    Оригинальная форма val_mask: {original_shape}")
        
        if len(original_shape) == 2:
            data.val_mask = data.val_mask.any(dim=1)
            print(f"    Исправленная форма val_mask: {data.val_mask.shape}")
    
    if hasattr(data, 'test_mask'):
        original_shape = data.test_mask.shape
        print(f"    Оригинальная форма test_mask: {original_shape}")
        
        if len(original_shape) == 2:
            data.test_mask = data.test_mask.any(dim=1)
            print(f"    Исправленная форма test_mask: {data.test_mask.shape}")
    
    return data

def fix_actor_masks(data):
    """
    Исправляет маски для Actor датасета
    
    В Actor датасете есть только train_mask и test_mask, нет val_mask
    """
    print(f"  Исправление масок для Actor датасета...")
    
    # Проверяем наличие и форму масок
    if hasattr(data, 'train_mask'):
        original_shape = data.train_mask.shape
        print(f"    Оригинальная форма train_mask: {original_shape}")
        
        # Actor маска может быть двумерной
        if len(original_shape) == 2:
            # Берем первый ненулевой элемент
            data.train_mask = data.train_mask[:, 0] if original_shape[1] > 0 else data.train_mask.any(dim=1)
            print(f"    Исправленная форма train_mask: {data.train_mask.shape}")
    
    if hasattr(data, 'test_mask'):
        original_shape = data.test_mask.shape
        print(f"    Оригинальная форма test_mask: {original_shape}")
        
        if len(original_shape) == 2:
            data.test_mask = data.test_mask[:, 0] if original_shape[1] > 0 else data.test_mask.any(dim=1)
            print(f"    Исправленная форма test_mask: {data.test_mask.shape}")
    
    # Добавляем val_mask, если его нет (берем часть из test_mask)
    if not hasattr(data, 'val_mask') or data.val_mask is None:
        print(f"    Создаю val_mask из test_mask...")
        num_nodes = data.num_nodes
        
        # Если есть test_mask, берем часть для валидации
        if hasattr(data, 'test_mask') and data.test_mask is not None:
            test_indices = torch.where(data.test_mask)[0]
            num_val = len(test_indices) // 3  # 1/3 test для val
            
            if num_val > 0:
                val_indices = test_indices[:num_val]
                data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                data.val_mask[val_indices] = True
                
                # Обновляем test_mask (убираем val из test)
                data.test_mask[val_indices] = False
                print(f"    Создан val_mask: {data.val_mask.sum().item()} узлов")
            else:
                # Создаем случайный val_mask
                data = create_random_split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        else:
            # Создаем случайное разбиение
            data = create_random_split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    return data

def fix_mask_dimensions(data):
    """
    Гарантирует, что все маски являются одномерными булевыми тензорами
    """
    mask_names = ['train_mask', 'val_mask', 'test_mask']
    
    for mask_name in mask_names:
        if hasattr(data, mask_name) and getattr(data, mask_name) is not None:
            mask = getattr(data, mask_name)
            
            # Проверяем размерность
            if mask.dim() > 1:
                print(f"  ⚠️ Маска {mask_name} имеет размерность {mask.dim()}. Преобразую в одномерную...")
                
                # Если это двумерный тензор с одной колонкой
                if mask.shape[1] == 1:
                    mask = mask.squeeze(1)
                # Если это двумерный тензор с несколькими колонками
                else:
                    # Предполагаем, что это one-hot кодировка
                    # Преобразуем в булев вектор (True если любой из классов True)
                    mask = mask.any(dim=1)
                
                setattr(data, mask_name, mask)
            
            # Гарантируем, что маска булевая
            if mask.dtype != torch.bool:
                print(f"  ⚠️ Маска {mask_name} имеет тип {mask.dtype}. Преобразую в bool...")
                setattr(data, mask_name, mask.bool())
    
    return data

def create_random_split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Создаёт случайное разбиение данных на train/val/test
    
    Args:
        data: Данные графа
        train_ratio: Доля обучающих узлов
        val_ratio: Доля валидационных узлов
        test_ratio: Доля тестовых узлов
    
    Returns:
        data: Данные с добавленными масками
    """
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    train_end = int(train_ratio * num_nodes)
    val_end = train_end + int(val_ratio * num_nodes)
    
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    data.train_mask[indices[:train_end]] = True
    data.val_mask[indices[train_end:val_end]] = True
    data.test_mask[indices[val_end:]] = True
    
    print(f"  Создано случайное разбиение:")
    print(f"    Train: {train_end}/{num_nodes} узлов")
    print(f"    Val: {val_end - train_end}/{num_nodes} узлов")
    print(f"    Test: {num_nodes - val_end}/{num_nodes} узлов")
    
    return data

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