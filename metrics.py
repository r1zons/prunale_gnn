# metrics.py
import torch
import time
import numpy as np

class PerformanceMetrics:
    """Класс для измерения производительности модели"""
    
    @staticmethod
    def measure_inference_time(model, data, num_runs=100, warmup=10):
        """
        Измеряет среднее время инференса
        
        Args:
            model: Модель GCN
            data: Данные графа
            num_runs: Количество прогонов для измерения
            warmup: Количество прогонов для прогрева
        
        Returns:
            Среднее время инференса в миллисекундах
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Прогрев
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(data.x, data.edge_index)
        
        # Измерение
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(data.x, data.edge_index)
                
                # Синхронизация для GPU/MPS
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        # Среднее время в миллисекундах
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        
        return avg_time, std_time
    
    @staticmethod
    def compute_accuracy(model, data):
        """
        Вычисляет точность на тестовом наборе
        
        Args:
            model: Модель GCN
            data: Данные графа
        
        Returns:
            Точность в диапазоне [0, 1]
        """
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            predictions = logits.argmax(dim=1)
            
            # Используем тестовую маску
            correct = predictions[data.test_mask] == data.y[data.test_mask]
            accuracy = correct.sum().item() / data.test_mask.sum().item()
        
        return accuracy
    
    @staticmethod
    def compute_model_size(model):
        """
        Вычисляет размер модели в мегабайтах
        
        Args:
            model: Модель GCN
        
        Returns:
            Размер модели в МБ
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.numel() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        
        return (param_size + buffer_size) / (1024 ** 2)
    
    @staticmethod
    def estimate_flops(model, data):
        """
        Оценивает количество операций с плавающей точкой
        
        Args:
            model: Модель GCN
            data: Данные графа
        
        Returns:
            Оценочное количество FLOPs
        """
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1)
        
        # Для GCNConv:
        # Первый слой: ~2 * num_edges * in_features + num_nodes * in_features * hidden_channels
        # Второй слой: ~2 * num_edges * hidden_channels + num_nodes * hidden_channels * out_features
        
        # Упрощенная оценка
        flops = 2 * num_edges * data.x.size(1) + num_nodes * data.x.size(1) * 16
        
        return flops

    @staticmethod
    def compute_effective_model_size(model):
        """Вычисляет эффективный размер модели"""
        
        total_params = 0
        nonzero_params = 0
        
        for param in model.parameters():
            if param.requires_grad:
                total_params += param.numel()
                nonzero = (param != 0).sum().item()
                nonzero_params += nonzero
        
        # Плотный формат: все параметры как float32
        dense_size = total_params * 4 / (1024 ** 2)  # МБ
        
        # Разреженный формат (COO): 
        # Для каждого ненулевого элемента: значение(4) + row_idx(4) + col_idx(4)
        # + заголовок матрицы (примерно 100 байт на слой)
        sparse_size = (nonzero_params * 12 + 200) / (1024 ** 2)  # МБ
        
        # Точка безубыточности: когда sparse_size < dense_size
        # Решаем неравенство: n*12 < N*4, где n = N*(1-s), s - разреженность
        # => (1-s)*12 < 4 => 1-s < 1/3 => s > 2/3 ≈ 0.666
        
        sparsity = 1 - (nonzero_params / total_params) if total_params > 0 else 0
        
        return {
            'dense_size_mb': dense_size,
            'sparse_size_mb': sparse_size,
            'compression_ratio': dense_size / sparse_size if sparse_size > 0 else 1,
            'total_params': total_params,
            'nonzero_params': nonzero_params,
            'sparsity': sparsity
        }