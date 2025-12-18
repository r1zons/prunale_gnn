# pruner.py (исправленная версия)
import torch
import torch.nn as nn

class StructuralPruner:
    """Реализует структурный прунинг (по столбцам) для GCN"""
    
    def __init__(self, sparsity_rate=0.5, method='l2_norm'):
        self.sparsity_rate = sparsity_rate
        self.method = method
        
    def compute_column_importance(self, weight, model=None, data=None):
        """Разные методы оценки важности столбцов"""
        
        if self.method == 'l2_norm':
            return torch.norm(weight, dim=0, p=2)
            
        elif self.method == 'gradient_importance':
            # Используем градиенты для оценки важности
            model.train()
            out = model(data.x, data.edge_index)
            loss = torch.nn.functional.nll_loss(
                out[data.train_mask], 
                data.y[data.train_mask]
            )
            loss.backward()
            
            # Важность = средний абсолютный градиент по столбцу
            with torch.no_grad():
                grad = model.conv1.lin.weight.grad
                importance = torch.mean(torch.abs(grad), dim=0)
            
            # Обнуляем градиенты
            model.zero_grad()
            return importance
            
        elif self.method == 'weight_magnitude':
            # Важность = средняя абсолютная величина веса
            return torch.mean(torch.abs(weight), dim=0)
            
        elif self.method == 'random':
            return torch.rand(weight.size(1))
    
    def prune_layer(self, weight, sparsity_rate=None):
        """Применяет прунинг к одному слою"""
        if sparsity_rate is None:
            sparsity_rate = self.sparsity_rate
        
        if sparsity_rate <= 0:
            return weight.clone(), torch.ones_like(weight, dtype=torch.bool)
        
        # Вычисляем важность столбцов
        importance = self.compute_column_importance(weight)
        
        # Определяем количество столбцов для удаления
        num_columns = weight.size(1)
        num_to_prune = int(sparsity_rate * num_columns)
        
        if num_to_prune <= 0:
            return weight.clone(), torch.ones_like(weight, dtype=torch.bool)
        
        # Находим наименее важные столбцы
        _, indices = torch.topk(importance, num_to_prune, largest=False)
        
        # Создаём маску
        mask = torch.ones_like(weight, dtype=torch.bool)
        mask[:, indices] = 0
        
        # Применяем маску
        pruned_weight = weight * mask.float()
        
        # Отладочная информация
        actual_sparsity = (mask == 0).sum().item() / mask.numel()
        print(f"    Целевая sparsity: {sparsity_rate:.3f}, фактическая: {actual_sparsity:.3f}")
        print(f"    Удалено столбцов: {num_to_prune}/{num_columns}")
        
        return pruned_weight, mask
    
    def prune_model(self, model, sparsity_rate=None):
        """Применяет прунинг ко всей модели"""
        if sparsity_rate is None:
            sparsity_rate = self.sparsity_rate
        
        masks = {}
        
        print(f"\n=== Применение структурного прунинга (sparsity={sparsity_rate}) ===")
        print("Доступные параметры в модели:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.size()}")
        
        # Применяем прунинг к весам свёрточных слоёв
        # В PyTorch Geometric GCNConv имеет параметры conv1.lin.weight, conv1.bias и т.д.
        for name, param in model.named_parameters():
            # Ищем веса линейных слоёв внутри GCNConv
            if ('lin.weight' in name) and ('conv1' in name or 'conv2' in name):
                print(f"\nПруним слой: {name}")
                print(f"  Размер весов: {param.size()}")
                
                # Пруним столбцы весовой матрицы
                pruned_weight, mask = self.prune_layer(param.data, sparsity_rate)
                
                # Сохраняем изменения
                param.data = pruned_weight
                masks[name] = mask
                model.save_pruning_mask(name, mask)
                
                # Проверяем результат
                zero_count = (pruned_weight == 0).sum().item()
                total_count = pruned_weight.numel()
                actual_sparsity = zero_count / total_count
                print(f"  Результат: {zero_count}/{total_count} нулей ({actual_sparsity:.3f})")
        
        return model, masks