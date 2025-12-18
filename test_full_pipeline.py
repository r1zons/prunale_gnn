# test_full_pipeline.py
import torch
import warnings
warnings.filterwarnings('ignore')

from config import CORR_CONFIG
from data_loader import load_dataset
from model import PrunableGCN
from pruner import StructuralPruner
from trainer import GNNTrainer
from metrics import PerformanceMetrics

def test_full_pipeline():
    print("=== ТЕСТ ПОЛНОГО ПАЙПЛАЙНА ===")
    
    # Конфигурация
    config = CORR_CONFIG
    config.epochs = 50
    config.hidden_channels = 16
    
    # 1. Загрузка данных
    data, dataset = load_dataset('Cora', device='cpu')
    
    # 2. Обучение модели
    print("\n1. Обучение модели...")
    model = PrunableGCN(
        in_channels=dataset.num_features,
        hidden_channels=config.hidden_channels,
        out_channels=dataset.num_classes
    )
    
    trainer = GNNTrainer(model, device='cpu')
    trainer.train(data, epochs=config.epochs, lr=config.lr, verbose=True)
    
    baseline_acc = trainer.evaluate(data)
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    
    # 3. Применение прунинга 50%
    print("\n2. Применение прунинга 50%...")
    sparsity = 0.5
    
    # Копируем модель
    import copy
    pruned_model = copy.deepcopy(model)
    
    # Применяем прунинг
    pruner = StructuralPruner(sparsity_rate=sparsity)
    pruned_model, masks = pruner.prune_model(pruned_model)
    pruned_model.apply_pruning_masks()
    
    # Проверяем прунинг
    print("\nПроверка прунинга:")
    report = pruned_model.get_detailed_sparsity_report()
    for layer_name, info in report.items():
        if layer_name != 'overall':
            sparsity_val = info['zero_params'] / info['total_params']
            print(f"  {layer_name}: {sparsity_val:.3f}")
    
    # 4. Дообучение
    print("\n3. Дообучение...")
    
    # Замораживаем нулевые веса
    pruned_model.freeze_pruned_weights()
    
    # Оптимизатор только для ненулевых весов
    optimizer = torch.optim.Adam(
        [p for p in pruned_model.parameters() if p.requires_grad],
        lr=config.lr * 0.1
    )
    
    for epoch in range(20):
        pruned_model.train()
        optimizer.zero_grad()
        out = pruned_model(data.x, data.edge_index)
        loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            pruned_model.eval()
            with torch.no_grad():
                out = pruned_model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
            print(f"  Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")
    
    # 5. Измерение метрик
    print("\n4. Измерение метрик...")
    
    # Точность
    pruned_model.eval()
    with torch.no_grad():
        out = pruned_model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        final_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    
    # Время инференса
    inference_time, _ = PerformanceMetrics.measure_inference_time(pruned_model, data, num_runs=20)
    
    # Размер модели
    size_info = PerformanceMetrics.compute_effective_model_size(pruned_model)
    
    print(f"\n=== РЕЗУЛЬТАТЫ ===")
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"Pruned accuracy:   {final_acc:.4f}")
    print(f"Accuracy drop:     {baseline_acc - final_acc:.4f}")
    print(f"Inference time:    {inference_time:.2f} ms")
    print(f"Dense model size:  {size_info['dense_size_mb']:.4f} MB")
    print(f"Sparse model size: {size_info['sparse_size_mb']:.4f} MB")
    print(f"Compression ratio: {size_info['compression_ratio']:.2f}x")
    print(f"Actual sparsity:   {size_info['sparsity']:.4f}")
    print(f"Target sparsity:   {sparsity:.4f}")

if __name__ == "__main__":
    test_full_pipeline()