# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class PrunableGCN(nn.Module):
    """
    GCN модель с поддержкой прунинга
    
    Args:
        in_channels: Размерность входных признаков
        hidden_channels: Размерность скрытого слоя
        out_channels: Количество классов
        dropout: Вероятность dropout
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        
        # Словарь для масок прунинга
        self.pruning_masks = {}
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def apply_pruning_masks(self):
        """Применяет маски прунинга к весам модели"""
        for name, param in self.named_parameters():
            if name in self.pruning_masks:
                param.data *= self.pruning_masks[name]
    
    def save_pruning_mask(self, layer_name, mask):
        """Сохраняет маску прунинга для слоя"""
        self.pruning_masks[layer_name] = mask
    
    def get_sparsity(self):
        """
        Вычисляет разреженность модели
        
        Returns:
            Доля нулевых весов во всех обучаемых параметрах
        """
        total_params = 0
        zero_params = 0
        
        for name, param in self.named_parameters():
            if 'weight' in name and param.requires_grad:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0
    
    def count_parameters(self):
        """Возвращает количество обучаемых параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self):
        """Вычисляет размер модели в мегабайтах"""
        param_size = 0
        for param in self.parameters():
            if param.requires_grad:
                param_size += param.numel() * param.element_size()
        
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        
        return (param_size + buffer_size) / (1024 ** 2)
    

    def get_detailed_sparsity(self):
        """Детальный подсчёт разреженности по слоям"""
        results = {}
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                total = param.numel()
                zeros = (param == 0).sum().item()
                sparsity = zeros / total if total > 0 else 0
                results[name] = {
                    'size': tuple(param.shape),
                    'total_params': total,
                    'zero_params': zeros,
                    'sparsity': sparsity
                }
        
        return results

    def print_sparsity_report(self):
        """Печатает отчёт о разреженности"""
        print("\n=== Отчёт о разреженности модели ===")
        total_params = 0
        total_zeros = 0
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                total = param.numel()
                zeros = (param == 0).sum().item()
                sparsity = zeros / total if total > 0 else 0
                
                print(f"  {name}: {tuple(param.shape)}")
                print(f"    Нулевых: {zeros}/{total} ({sparsity:.3f})")
                
                total_params += total
                total_zeros += zeros
        
        if total_params > 0:
            overall_sparsity = total_zeros / total_params
            print(f"\n  Общая разреженность: {total_zeros}/{total_params} ({overall_sparsity:.4f})")
        
        return overall_sparsity
    
    # В model.py, добавьте в класс PrunableGCN:

    def apply_pruning_masks(self):
        """Применяет маски прунинга к весам модели (для гарантии)"""
        for name, param in self.named_parameters():
            if name in self.pruning_masks:
                param.data *= self.pruning_masks[name]

    def get_detailed_sparsity_report(self):
        """Возвращает детальный отчёт о разреженности по слоям"""
        report = {}
        
        for name, param in self.named_parameters():
            if 'weight' in name and param.requires_grad:
                total = param.numel()
                zeros = (param == 0).sum().item()
                sparsity = zeros / total if total > 0 else 0
                report[name] = {
                    'shape': tuple(param.shape),
                    'total_params': total,
                    'zero_params': zeros,
                    'sparsity': sparsity
                }
        
        # Общая статистика
        total_all = sum(info['total_params'] for info in report.values())
        zeros_all = sum(info['zero_params'] for info in report.values())
        
        if total_all > 0:
            report['overall'] = {
                'total_params': total_all,
                'zero_params': zeros_all,
                'sparsity': zeros_all / total_all
            }
        
        return report
    

    def freeze_pruned_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                # Создаём маску: 1 для ненулевых весов, 0 для нулевых
                mask = (param != 0).float()
                # Регистрируем хук для градиентов
                def hook(grad, mask=mask):
                    return grad * mask
                if param.requires_grad:
                    param.register_hook(hook)
