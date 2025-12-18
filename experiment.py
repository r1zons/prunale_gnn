# experiment.py
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, List
import json
import os
from datetime import datetime

from config import ExperimentConfig, DEFAULT_SPARSITY_RATES
from data_loader import load_dataset, get_device
from model import PrunableGCN
from pruner import StructuralPruner
from trainer import GNNTrainer
from metrics import PerformanceMetrics

class GNNPruningExperiment:
    """Основной класс для проведения экспериментов по прунингу"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = get_device(config.use_mps)
        self.results = []
        
        print(f"=== Настройка эксперимента ===")
        print(f"Датасет: {config.dataset_name}")
        print(f"Устройство: {self.device}")
        print(f"Скрытый размер: {config.hidden_channels}")
        print(f"Эпох: {config.epochs}")
        print(f"Уровни разреженности: {DEFAULT_SPARSITY_RATES}")
    
    # В методе run_single_experiment исправляем:

    def run_single_experiment(self, sparsity_rate, run_id=0):
        """Запускает один эксперимент с заданным уровнем разреженности"""
        
        print(f"\n{'='*60}")
        print(f"ЗАПУСК {run_id+1}: Sparsity={sparsity_rate:.2f}")
        print(f"{'='*60}")
        
        # 1. Загрузка данных
        data, dataset = load_dataset(self.config.dataset_name, self.device)
        
        # 2. Создание модели
        model = PrunableGCN(
            in_channels=dataset.num_features,
            hidden_channels=self.config.hidden_channels,
            out_channels=dataset.num_classes,
            dropout=self.config.dropout
        )
        
        # 3. Обучение baseline
        print("\n1. Обучение модели...")
        trainer = GNNTrainer(model, self.device)
        trainer.train(
            data, 
            epochs=self.config.epochs,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            verbose=True
        )
        
        baseline_acc = trainer.evaluate(data)
        print(f"Точность после обучения: {baseline_acc:.4f}")
        
        # 4. Сохраняем обученные веса ДО прунинга
        trained_state_dict = model.state_dict().copy()
        
        # 5. Применение прунинга (если sparsity > 0)
        if sparsity_rate > 0:
            print(f"\n2. Применение структурного прунинга ({sparsity_rate*100:.0f}%)...")
            
            # Создаём новую модель с теми же обученными весами
            pruned_model = PrunableGCN(
                in_channels=dataset.num_features,
                hidden_channels=self.config.hidden_channels,
                out_channels=dataset.num_classes,
                dropout=self.config.dropout
            )
            pruned_model.load_state_dict(trained_state_dict)
            
            # Применяем прунинг
            pruner = StructuralPruner(sparsity_rate=sparsity_rate, method='l2_norm')
            pruned_model, masks = pruner.prune_model(pruned_model)
            
            # ПРИНУДИТЕЛЬНО применяем маски (важно!)
            pruned_model.apply_pruning_masks()
            
            # Проверяем результат прунинга
            print("\nПроверка прунинга:")
            report = pruned_model.get_detailed_sparsity_report()
            for layer_name, info in report.items():
                if layer_name != 'overall':
                    sparsity = info['zero_params'] / info['total_params']
                    print(f"  {layer_name}: {sparsity:.3f}")
            
            # Заменяем модель на пруненую
            model = pruned_model
            
            # 6. Дообучение после прунинга с ЗАМОРАЖИВАНИЕМ обнулённых весов
            # В experiment.py улучшаем дообучение:
            if self.config.retrain_after_prune:
                print(f"\n3. Дообучение после прунинга...")
                
                # Используем меньший learning rate и scheduler
                optimizer = torch.optim.Adam(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=self.config.lr * 0.01,  # Ещё меньше LR
                    weight_decay=self.config.weight_decay
                )
                
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', patience=10, factor=0.5
                )
                
                best_acc = 0
                patience_counter = 0
                max_patience = 30
                
                for epoch in range(100):  # Увеличиваем до 100 эпох
                    model.train()
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index)
                    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                    
                    # Зануляем градиенты обнулённых весов
                    for name, param in model.named_parameters():
                        if name in model.pruning_masks:
                            mask = model.pruning_masks[name]
                            if param.grad is not None:
                                param.grad *= mask.float()
                    
                    optimizer.step()
                    
                    # Оценка
                    if epoch % 5 == 0:
                        model.eval()
                        with torch.no_grad():
                            out = model(data.x, data.edge_index)
                            pred = out.argmax(dim=1)
                            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
                            
                            print(f"  Epoch {epoch:3d}: loss={loss:.4f}, val_acc={val_acc:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
                            
                            scheduler.step(val_acc)
                            
                            # Early stopping
                            if val_acc > best_acc:
                                best_acc = val_acc
                                patience_counter = 0
                            else:
                                patience_counter += 1
                                
                            if patience_counter >= max_patience:
                                print(f"  Early stopping at epoch {epoch}")
                                break
        
        # 7. Измерение метрик
        print(f"\n4. Измерение метрик...")
        
        # Точность
        trainer = GNNTrainer(model, self.device)
        accuracy = trainer.evaluate(data)
        
        # Время инференса
        inference_time, inference_std = PerformanceMetrics.measure_inference_time(
            model, data, num_runs=50
        )
        
        # Размер модели (в плотном формате)
        model_size_mb = model.get_model_size_mb()
        
        # Разреженность
        actual_sparsity = model.get_sparsity()
        
        # FLOPs (оценочно)
        flops = PerformanceMetrics.estimate_flops(model, data)
        
        # Количество параметров
        num_params = model.count_parameters()
        
        # 8. Сохранение результатов
        result = {
            'run_id': run_id,
            'sparsity_target': sparsity_rate,
            'sparsity_actual': actual_sparsity,
            'accuracy': accuracy,
            'baseline_accuracy': baseline_acc,
            'inference_time_ms': inference_time,
            'inference_std_ms': inference_std,
            'model_size_mb': model_size_mb,
            'flops': flops,
            'num_params': num_params,
            'hidden_channels': self.config.hidden_channels,
            'dataset': self.config.dataset_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\nРезультаты:")
        print(f"  Точность: {accuracy:.4f} (baseline: {baseline_acc:.4f})")
        print(f"  Время инференса: {inference_time:.2f} ± {inference_std:.2f} мс")
        print(f"  Размер модели: {model_size_mb:.2f} МБ")
        print(f"  Фактическая разреженность: {actual_sparsity:.4f}")
        
        return result
    
    # В методе run_full_experiment класса GNNPruningExperiment
    def run_full_experiment(self):
        """Запускает полный эксперимент со всеми уровнями разреженности"""
        
        print(f"\n{'='*50}")
        print(f"ЗАПУСК ПОЛНОГО ЭКСПЕРИМЕНТА")
        print(f"{'='*50}")
        
        all_results = []
        
        for sparsity in DEFAULT_SPARSITY_RATES:
            # Запускаем несколько раз для статистики
            for run in range(self.config.runs):
                result = self.run_single_experiment(sparsity, run)
                all_results.append(result)
        
        # Сохраняем результаты в DataFrame
        self.df_results = pd.DataFrame(all_results)
        
        # Агрегируем результаты по уровню разреженности
        # Более простой способ для избежания MultiIndex
        aggregated = []
        for sparsity in self.df_results['sparsity_target'].unique():
            mask = self.df_results['sparsity_target'] == sparsity
            subset = self.df_results[mask]
            
            aggregated.append({
                'sparsity_target': sparsity,
                'accuracy_mean': subset['accuracy'].mean(),
                'accuracy_std': subset['accuracy'].std(),
                'inference_time_ms_mean': subset['inference_time_ms'].mean(),
                'inference_time_ms_std': subset['inference_time_ms'].std(),
                'model_size_mb_mean': subset['model_size_mb'].mean(),
                'sparsity_actual_mean': subset['sparsity_actual'].mean()
            })
        
        self.aggregated_results = pd.DataFrame(aggregated)
        
        print(f"\n{'='*50}")
        print(f"ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
        print(f"{'='*50}")
        
        return self.df_results, self.aggregated_results

    def save_results(self, output_dir='./results'):
        """Сохраняет результаты экспериментов"""
        
        if not hasattr(self, 'df_results'):
            print("Нет результатов для сохранения. Сначала запустите эксперимент.")
            return
        
        # Создаем директорию, если её нет
        os.makedirs(output_dir, exist_ok=True)
        
        # Генерируем имя файла на основе даты и датасета
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_base = f"{self.config.dataset_name}_{timestamp}"
        
        # Сохраняем полные результаты
        full_path = os.path.join(output_dir, f"{filename_base}_full.csv")
        self.df_results.to_csv(full_path, index=False)
        
        # Сохраняем агрегированные результаты
        agg_path = os.path.join(output_dir, f"{filename_base}_aggregated.csv")
        self.aggregated_results.to_csv(agg_path)
        
        # Сохраняем конфигурацию
        config_path = os.path.join(output_dir, f"{filename_base}_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        print(f"\nРезультаты сохранены в {output_dir}:")
        print(f"  Полные результаты: {full_path}")
        print(f"  Агрегированные: {agg_path}")
        print(f"  Конфигурация: {config_path}")
        
        return full_path, agg_path, config_path