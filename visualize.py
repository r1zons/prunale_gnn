# visualize.py (исправленная версия)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

def plot_experiment_results(df_results: pd.DataFrame, 
                           aggregated_df: Optional[pd.DataFrame] = None,
                           save_path: str = None):
    """
    Создает визуализации результатов экспериментов
    
    Args:
        df_results: DataFrame с полными результатами
        aggregated_df: DataFrame с агрегированными результатами
        save_path: Путь для сохранения графиков
    """
    
    # Настройка стиля
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Проверяем структуру aggregated_df
    if aggregated_df is not None:
        print(f"Структура aggregated_df: {aggregated_df.shape}")
        print(f"Колонки aggregated_df: {aggregated_df.columns.tolist()}")
    
    if aggregated_df is None:
        # Агрегируем сами, если не предоставлено
        aggregated_df = df_results.groupby('sparsity_target').agg({
            'accuracy': ['mean', 'std'],
            'inference_time_ms': ['mean', 'std'],
            'model_size_mb': 'mean'
        }).reset_index()
        
        # Преобразуем MultiIndex в простые названия
        aggregated_df.columns = ['sparsity', 'accuracy_mean', 'accuracy_std',
                                'time_mean', 'time_std', 'size_mean']
    else:
        # У aggregated_df уже есть MultiIndex после groupby
        # Нужно преобразовать его в плоскую структуру
        if isinstance(aggregated_df.columns, pd.MultiIndex):
            aggregated_df = aggregated_df.copy()
            # Объединяем уровни MultiIndex
            aggregated_df.columns = ['_'.join(filter(None, col)).rstrip('_') 
                                   for col in aggregated_df.columns.values]
            
            # Проверяем, есть ли индекс sparsity_target
            if aggregated_df.index.name == 'sparsity_target':
                aggregated_df = aggregated_df.reset_index()
                aggregated_df.rename(columns={'sparsity_target': 'sparsity'}, inplace=True)
            elif 'sparsity_target' in aggregated_df.columns:
                aggregated_df.rename(columns={'sparsity_target': 'sparsity'}, inplace=True)
        
        # Переименовываем колонки для единообразия
        column_mapping = {}
        if 'accuracy_mean' not in aggregated_df.columns and 'accuracy' in aggregated_df.columns:
            # Если есть только 'accuracy', это среднее значение
            column_mapping['accuracy'] = 'accuracy_mean'
        
        if 'inference_time_ms_mean' in aggregated_df.columns:
            column_mapping['inference_time_ms_mean'] = 'time_mean'
        if 'inference_time_ms_std' in aggregated_df.columns:
            column_mapping['inference_time_ms_std'] = 'time_std'
        
        if column_mapping:
            aggregated_df = aggregated_df.rename(columns=column_mapping)
    
    # Проверяем наличие необходимых колонок
    required_columns = ['sparsity', 'accuracy_mean', 'time_mean', 'size_mean']
    for col in required_columns:
        if col not in aggregated_df.columns:
            print(f"Предупреждение: отсутствует колонка {col}")
            print(f"Доступные колонки: {aggregated_df.columns.tolist()}")
            # Попробуем найти альтернативные названия
            if col == 'accuracy_mean' and 'accuracy' in aggregated_df.columns:
                aggregated_df['accuracy_mean'] = aggregated_df['accuracy']
            if col == 'time_mean' and 'inference_time_ms' in aggregated_df.columns:
                aggregated_df['time_mean'] = aggregated_df['inference_time_ms']
    
    # Создаем фигуру с 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    try:
        # 1. Точность vs Разреженность
        ax = axes[0, 0]
        if 'accuracy_std' in aggregated_df.columns:
            ax.errorbar(aggregated_df['sparsity'], aggregated_df['accuracy_mean'],
                       yerr=aggregated_df['accuracy_std'], 
                       marker='o', markersize=8, capsize=5, linewidth=2)
        else:
            ax.plot(aggregated_df['sparsity'], aggregated_df['accuracy_mean'],
                   marker='o', markersize=8, linewidth=2)
        ax.set_xlabel('Уровень разреженности', fontsize=12)
        ax.set_ylabel('Точность', fontsize=12)
        ax.set_title('Влияние прунинга на точность модели', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        
        # 2. Время инференса vs Разреженность
        ax = axes[0, 1]
        if 'time_std' in aggregated_df.columns:
            ax.errorbar(aggregated_df['sparsity'], aggregated_df['time_mean'],
                       yerr=aggregated_df['time_std'],
                       marker='s', markersize=8, capsize=5, linewidth=2, color='red')
        else:
            ax.plot(aggregated_df['sparsity'], aggregated_df['time_mean'],
                   marker='s', markersize=8, linewidth=2, color='red')
        ax.set_xlabel('Уровень разреженности', fontsize=12)
        ax.set_ylabel('Время инференса (мс)', fontsize=12)
        ax.set_title('Влияние прунинга на скорость инференса', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        
        # 3. Размер модели vs Разреженность
        ax = axes[1, 0]
        ax.plot(aggregated_df['sparsity'], aggregated_df['size_mean'],
                marker='^', markersize=8, linewidth=2, color='green')
        ax.set_xlabel('Уровень разреженности', fontsize=12)
        ax.set_ylabel('Размер модели (МБ)', fontsize=12)
        ax.set_title('Влияние прунинга на размер модели', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        
        # 4. Trade-off: Точность vs Время инференса
        ax = axes[1, 1]
        scatter = ax.scatter(aggregated_df['time_mean'], aggregated_df['accuracy_mean'],
                            c=aggregated_df['sparsity'], s=200, cmap='viridis', alpha=0.8)
        
        # Добавляем аннотации с уровнем разреженности
        for i, row in aggregated_df.iterrows():
            ax.annotate(f"{row['sparsity']:.1f}", 
                       (row['time_mean'], row['accuracy_mean']),
                       fontsize=9, ha='center', va='center', color='white',
                       fontweight='bold')
        
        ax.set_xlabel('Время инференса (мс)', fontsize=12)
        ax.set_ylabel('Точность', fontsize=12)
        ax.set_title('Точность vs Скорость (trade-off)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Цветовая шкала для разреженности
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Уровень разреженности', fontsize=10)
        
        # Общая настройка
        dataset_name = df_results['dataset'].iloc[0] if 'dataset' in df_results.columns else 'Unknown'
        hidden_size = df_results['hidden_channels'].iloc[0] if 'hidden_channels' in df_results.columns else 'Unknown'
        
        plt.suptitle(f'Результаты экспериментов по структурному прунингу GNN\n'
                    f'Датасет: {dataset_name}, '
                    f'Скрытый размер: {hidden_size}', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Сохранение или отображение
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Графики сохранены в {save_path}")
        else:
            plt.show()
        
    except KeyError as e:
        print(f"Ошибка при построении графиков: отсутствует колонка {e}")
        print(f"Доступные колонки: {aggregated_df.columns.tolist()}")
        # Выводим aggregated_df для отладки
        print("\nСодержимое aggregated_df:")
        print(aggregated_df)
    
    return fig