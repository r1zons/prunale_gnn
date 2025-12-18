# quick_fix_experiment.py
import torch
import pandas as pd
from config import CORR_CONFIG
from experiment import GNNPruningExperiment

# Используем только умеренные уровни прунинга
config = CORR_CONFIG
config.epochs = 150  # Больше эпох обучения
config.lr = 0.005  # Меньше learning rate
config.retrain_after_prune = True
config.runs = 3

# Только разумные уровни прунинга
sparsities_to_test = [0.0, 0.1, 0.2, 0.3, 0.4]

print("=== ЗАПУСК ОПТИМИЗИРОВАННОГО ЭКСПЕРИМЕНТА ===")
print(f"Уровни прунинга: {sparsities_to_test}")
print(f"Эпох обучения: {config.epochs}")
print(f"Learning rate: {config.lr}")

experiment = GNNPruningExperiment(config)

# Модифицируем эксперимент для тестирования только нужных уровней
all_results = []
for sparsity in sparsities_to_test:
    for run in range(config.runs):
        result = experiment.run_single_experiment(sparsity, run)
        all_results.append(result)

# Сохраняем результаты
df = pd.DataFrame(all_results)
df.to_csv('optimized_results.csv', index=False)

# Анализ
print("\n=== АНАЛИЗ РЕЗУЛЬТАТОВ ===")
baseline = df[df['sparsity_target'] == 0.0]['accuracy'].mean()
print(f"Baseline accuracy: {baseline:.4f}")

for sparsity in [0.1, 0.2, 0.3, 0.4]:
    mask = df['sparsity_target'] == sparsity
    if mask.any():
        avg_acc = df[mask]['accuracy'].mean()
        avg_time = df[mask]['inference_time_ms'].mean()
        acc_drop = baseline - avg_acc
        print(f"\nSparsity {sparsity:.1f}:")
        print(f"  Accuracy: {avg_acc:.4f} (drop: {acc_drop:.4f})")
        print(f"  Inference time: {avg_time:.2f} ms")
        print(f"  Acceptable: {'YES' if acc_drop < 0.05 else 'NO'}")