# test_simple.py
import torch
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric')

from config import CORR_CONFIG
from experiment import GNNPruningExperiment

# Простая конфигурация для теста
config = CORR_CONFIG
config.epochs = 50  # Меньше для быстрого теста
config.runs = 1  # Только один запуск
config.hidden_channels = 16

print("=== ЗАПУСК ПРОСТОГО ТЕСТА ===")
experiment = GNNPruningExperiment(config)

# Запускаем только несколько уровней разреженности для теста
test_sparsities = [0.0, 0.3, 0.5]
print(f"Тестируемые уровни разреженности: {test_sparsities}")

all_results = []
for sparsity in test_sparsities:
    result = experiment.run_single_experiment(sparsity, run_id=0)
    all_results.append(result)

# Сохраняем результаты
import pandas as pd
df = pd.DataFrame(all_results)
print("\n=== РЕЗУЛЬТАТЫ ===")
print(df[['sparsity_target', 'accuracy', 'inference_time_ms', 'model_size_mb']])

# Простая визуализация без сложной обработки
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(df['sparsity_target'], df['accuracy'], 'bo-')
plt.xlabel('Уровень разреженности')
plt.ylabel('Точность')
plt.title('Точность vs Разреженность')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(df['sparsity_target'], df['inference_time_ms'], 'ro-')
plt.xlabel('Уровень разреженности')
plt.ylabel('Время инференса (мс)')
plt.title('Время инференса vs Разреженность')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_test_results.png', dpi=150)
plt.show()

print(f"\nРезультаты сохранены в simple_test_results.png")
print("Тест завершен успешно!")