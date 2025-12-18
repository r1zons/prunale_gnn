# debug_pruning.py (обновлённый)
import torch
from torch_geometric.datasets import Planetoid

from model import PrunableGCN
from pruner import StructuralPruner

def test_pruning():
    # Загружаем данные
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]
    
    # Создаём модель
    model = PrunableGCN(
        in_channels=dataset.num_features,
        hidden_channels=16,
        out_channels=dataset.num_classes
    )
    
    print("=== Тестирование прунинга ===")
    
    # Проверяем исходное состояние
    print("\n1. Исходная модель:")
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"  {name}: {param.size()}, ненулевых: {(param != 0).sum().item()}/{param.numel()}")
    
    sparsity_rate = 0.5
    
    # Применяем прунинг
    print(f"\n2. Применяем прунинг с sparsity={sparsity_rate}")
    pruner = StructuralPruner(sparsity_rate=sparsity_rate)
    pruned_model, masks = pruner.prune_model(model)
    
    # Применяем маски для гарантии
    pruned_model.apply_pruning_masks()
    
    # Проверяем результат
    print("\n3. Модель после прунинга:")
    report = pruned_model.get_detailed_sparsity_report()
    
    for layer_name, info in report.items():
        if layer_name != 'overall':
            print(f"  {layer_name}: {info['shape']}")
            print(f"    Нулей: {info['zero_params']}/{info['total_params']} ({info['sparsity']:.3f})")
    
    if 'overall' in report:
        info = report['overall']
        print(f"\n4. Общая статистика:")
        print(f"  Всего параметров: {info['total_params']}")
        print(f"  Нулевых параметров: {info['zero_params']}")
        print(f"  Фактическая разреженность: {info['sparsity']:.4f}")
        print(f"  Целевая разреженность: {sparsity_rate}")
    
    # Проверяем метод get_sparsity
    print(f"\n5. Проверка model.get_sparsity(): {pruned_model.get_sparsity():.4f}")
    
    # Проверяем маски
    print(f"\n6. Количество сохранённых масок: {len(pruned_model.pruning_masks)}")
    for name, mask in pruned_model.pruning_masks.items():
        print(f"  Маска для {name}: размер {mask.size()}, "
              f"нулей в маске: {(mask == 0).sum().item()}/{mask.numel()}")

if __name__ == "__main__":
    test_pruning()