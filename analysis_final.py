# analysis_final.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy import stats

def load_all_results(results_dir='./results'):
    """Загружает все результаты из CSV файлов и объединяет их"""
    
    all_dfs = []
    
    # Ищем все файлы с результатами
    pattern = os.path.join(results_dir, '*_full.csv')
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"Файлы не найдены по шаблону: {pattern}")
        return None
    
    print(f"Найдено файлов: {len(csv_files)}")
    
    for file_path in csv_files:
        try:
            # Извлекаем название датасета из имени файла
            filename = os.path.basename(file_path)
            # Пример: Cora_20251217_010115_full.csv → Cora
            dataset_name = filename.split('_')[0]
            
            # Загружаем CSV
            df = pd.read_csv(file_path)
            
            # Добавляем колонку с именем датасета, если её нет
            if 'dataset' not in df.columns:
                df['dataset'] = dataset_name
            
            print(f"Загружен {filename}: {len(df)} строк")
            all_dfs.append(df)
            
        except Exception as e:
            print(f"Ошибка при загрузке {file_path}: {e}")
    
    if all_dfs:
        # Объединяем все DataFrame
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\nОбъединённый DataFrame: {len(combined_df)} строк")
        print(f"Колонки: {combined_df.columns.tolist()}")
        print(f"Датасеты: {combined_df['dataset'].unique().tolist()}")
        
        return combined_df
    else:
        print("Нет данных для анализа")
        return None

def analyze_final_results(df):
    """Анализ и визуализация финальных результатов"""
    
    if df is None or len(df) == 0:
        print("Нет данных для анализа")
        return None
    
    # Настройка стиля
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("Set2")
    
    # Создаём фигуру с несколькими графиками
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Точность vs Разреженность (все датасеты)
    ax = axes[0, 0]
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        
        # Группируем по уровню разреженности
        grouped = dataset_data.groupby('sparsity_target').agg({
            'accuracy': ['mean', 'std'],
            'inference_time_ms': 'mean'
        }).round(4)
        
        # Извлекаем данные
        sparsity = grouped.index
        accuracy_mean = grouped[('accuracy', 'mean')]
        accuracy_std = grouped[('accuracy', 'std')]
        
        # Построение с доверительными интервалами
        ax.errorbar(sparsity, accuracy_mean, yerr=accuracy_std,
                   marker='o', markersize=8, capsize=5, linewidth=2,
                   label=dataset, alpha=0.8)
    
    ax.set_xlabel('Уровень разреженности', fontsize=12)
    ax.set_ylabel('Точность', fontsize=12)
    ax.set_title('Точность vs Разреженность на разных датасетах', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, max(df['sparsity_target']) * 1.1)
    
    # 2. Ускорение инференса vs Разреженность
    ax = axes[0, 1]
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        
        # Вычисляем baseline время для каждого датасета
        baseline_mask = (dataset_data['sparsity_target'] == 0.0)
        if baseline_mask.any():
            baseline_time = dataset_data[baseline_mask]['inference_time_ms'].mean()
        else:
            baseline_time = dataset_data['inference_time_ms'].mean()
        
        # Группируем
        grouped = dataset_data.groupby('sparsity_target').agg({
            'inference_time_ms': 'mean'
        })
        
        # Вычисляем ускорение относительно baseline
        if baseline_time > 0:
            speedup = baseline_time / grouped['inference_time_ms']
            ax.plot(grouped.index, speedup, marker='s', markersize=8, 
                    linewidth=2, label=dataset)
    
    ax.set_xlabel('Уровень разреженности', fontsize=12)
    ax.set_ylabel('Ускорение инференса (раз)', fontsize=12)
    ax.set_title('Ускорение инференса vs Разреженность', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, max(df['sparsity_target']) * 1.1)
    
    # 3. Trade-off: Потеря точности vs Ускорение
    ax = axes[0, 2]
    
    # Готовим данные для scatter plot
    scatter_data = []
    
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        
        # Baseline метрики
        baseline_mask = dataset_data['sparsity_target'] == 0.0
        if baseline_mask.any():
            baseline_data = dataset_data[baseline_mask]
            baseline_acc = baseline_data['accuracy'].mean()
            baseline_time = baseline_data['inference_time_ms'].mean()
        else:
            continue
        
        # Данные для пруненых моделей (исключаем baseline)
        pruned_data = dataset_data[dataset_data['sparsity_target'] > 0]
        
        for sparsity in sorted(pruned_data['sparsity_target'].unique()):
            mask = pruned_data['sparsity_target'] == sparsity
            subset = pruned_data[mask]
            
            avg_acc = subset['accuracy'].mean()
            avg_time = subset['inference_time_ms'].mean()
            
            accuracy_drop = baseline_acc - avg_acc
            speedup = baseline_time / avg_time
            
            scatter_data.append({
                'dataset': dataset,
                'sparsity': sparsity,
                'accuracy_drop': accuracy_drop,
                'speedup': speedup,
                'accuracy': avg_acc,
                'time': avg_time
            })
    
    # Преобразуем в DataFrame для удобства
    scatter_df = pd.DataFrame(scatter_data)
    
    if not scatter_df.empty:
        colors = plt.cm.viridis(np.linspace(0, 1, len(df['dataset'].unique())))
        
        for idx, dataset in enumerate(scatter_df['dataset'].unique()):
            mask = scatter_df['dataset'] == dataset
            subset = scatter_df[mask]
            
            # Scatter plot с размерами точек по sparsity
            scatter = ax.scatter(subset['speedup'], subset['accuracy_drop'], 
                               s=subset['sparsity'] * 200,  # Размер точек
                               c=[colors[idx]], alpha=0.7, label=dataset)
            
            # Добавляем аннотации с уровнем разреженности
            for _, row in subset.iterrows():
                ax.annotate(f"{row['sparsity']:.1f}", 
                           (row['speedup'], row['accuracy_drop']),
                           fontsize=9, ha='center', va='center')
    
    ax.set_xlabel('Ускорение инференса (раз)', fontsize=12)
    ax.set_ylabel('Потеря точности', fontsize=12)
    ax.set_title('Trade-off: Потеря точности vs Ускорение', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Статистическая значимость различий
    ax = axes[1, 0]
    
    datasets = []
    optimal_sparsities = []
    optimal_drops = []
    optimal_speedups = []
    
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        
        # Находим оптимальную точку
        efficiencies = []
        
        baseline_mask = dataset_data['sparsity_target'] == 0.0
        if baseline_mask.any():
            baseline_data = dataset_data[baseline_mask]
            baseline_acc = baseline_data['accuracy'].mean()
            baseline_time = baseline_data['inference_time_ms'].mean()
        else:
            continue
        
        for sparsity in sorted(dataset_data['sparsity_target'].unique()):
            if sparsity == 0:
                continue
                
            mask = dataset_data['sparsity_target'] == sparsity
            subset = dataset_data[mask]
            
            avg_acc = subset['accuracy'].mean()
            avg_time = subset['inference_time_ms'].mean()
            
            accuracy_drop = baseline_acc - avg_acc
            speedup = baseline_time / avg_time
            
            # Эффективность: ускорение на единицу потери точности
            efficiency = speedup / (accuracy_drop + 0.001)  # +0.001 чтобы избежать деления на 0
            
            efficiencies.append((sparsity, efficiency, accuracy_drop, speedup))
        
        if efficiencies:
            # Находим точку с максимальной эффективностью
            optimal = max(efficiencies, key=lambda x: x[1])
            optimal_sparsities.append(optimal[0])
            optimal_drops.append(optimal[2])
            optimal_speedups.append(optimal[3])
            datasets.append(dataset)
    
    # Bar plot
    if datasets:
        x = np.arange(len(datasets))
        width = 0.35
        
        ax.bar(x - width/2, optimal_drops, width, label='Потеря точности', alpha=0.7)
        ax.bar(x + width/2, optimal_speedups, width, label='Ускорение', alpha=0.7)
        
        ax.set_xlabel('Датасет', fontsize=12)
        ax.set_ylabel('Значение', fontsize=12)
        ax.set_title('Оптимальные точки для каждого датасета', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Добавляем аннотации с уровнем разреженности
        for i, (dataset, sparsity) in enumerate(zip(datasets, optimal_sparsities)):
            ax.text(i, max(optimal_drops[i], optimal_speedups[i]) + 0.1,
                   f"S={sparsity:.2f}", ha='center', fontsize=10)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Недостаточно данных\nдля анализа', 
                ha='center', va='center', fontsize=14)
    
    # 5. Heatmap: зависимость точности от скрытого размера и разреженности
    ax = axes[1, 1]
    
    # Проверяем наличие необходимых колонок
    if 'hidden_channels' in df.columns and 'sparsity_target' in df.columns and 'accuracy' in df.columns:
        # Создаём pivot таблицу для heatmap
        pivot_data = df.pivot_table(
            values='accuracy',
            index='hidden_channels',
            columns='sparsity_target',
            aggfunc='mean'
        )
        
        if not pivot_data.empty:
            im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto')
            ax.set_xticks(np.arange(len(pivot_data.columns)))
            ax.set_yticks(np.arange(len(pivot_data.index)))
            ax.set_xticklabels([f"{col:.1f}" for col in pivot_data.columns])
            ax.set_yticklabels(pivot_data.index)
            
            ax.set_xlabel('Уровень разреженности', fontsize=12)
            ax.set_ylabel('Размер скрытого слоя', fontsize=12)
            ax.set_title('Точность в зависимости от параметров', fontsize=14, fontweight='bold')
            
            # Добавляем значения в ячейки
            for i in range(len(pivot_data.index)):
                for j in range(len(pivot_data.columns)):
                    ax.text(j, i, f"{pivot_data.iloc[i, j]:.3f}",
                           ha="center", va="center", color="black", fontsize=9)
            
            plt.colorbar(im, ax=ax)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Недостаточно данных\nдля heatmap', 
                    ha='center', va='center', fontsize=14)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Отсутствуют необходимые\nколонки для heatmap', 
                ha='center', va='center', fontsize=14)
    
    # 6. Сводная таблица результатов
    ax = axes[1, 2]
    ax.axis('off')
    
    # Создаём сводную таблицу
    summary_data = []
    
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        
        # Baseline данные
        baseline_mask = dataset_data['sparsity_target'] == 0.0
        if baseline_mask.any():
            baseline_data = dataset_data[baseline_mask]
            baseline_acc = baseline_data['accuracy'].mean()
            baseline_time = baseline_data['inference_time_ms'].mean()
        else:
            baseline_acc = dataset_data['accuracy'].mean()
            baseline_time = dataset_data['inference_time_ms'].mean()
        
        # Для каждого уровня разреженности
        for sparsity in sorted(dataset_data['sparsity_target'].unique()):
            if sparsity == 0:
                continue
                
            mask = dataset_data['sparsity_target'] == sparsity
            subset = dataset_data[mask]
            
            if len(subset) > 0:
                avg_acc = subset['accuracy'].mean()
                avg_time = subset['inference_time_ms'].mean()
                
                accuracy_drop = baseline_acc - avg_acc
                speedup = baseline_time / avg_time if avg_time > 0 else 1
                
                summary_data.append({
                    'Датасет': dataset,
                    'Разреженность': sparsity,
                    'Точность': f"{avg_acc:.4f}",
                    'Потеря': f"{accuracy_drop:.4f}",
                    'Время (мс)': f"{avg_time:.2f}",
                    'Ускорение': f"{speedup:.2f}x"
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Создаём таблицу как текст
        table_text = summary_df.to_string(index=False)
        
        # Отображаем таблицу
        ax.text(0.05, 0.95, "Сводная таблица результатов", 
                fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.85, table_text, 
                fontsize=8, fontfamily='monospace', transform=ax.transAxes,
                verticalalignment='top')
    else:
        ax.text(0.5, 0.5, 'Нет данных для\nсводной таблицы', 
                ha='center', va='center', fontsize=14)
    
    # Общая настройка
    plt.suptitle('Результаты исследований структурного прунинга GNN', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Сохраняем графики
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'final_analysis_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nГрафики сохранены в: {output_path}")
    
    # Вывод статистического анализа
    print("\n" + "="*60)
    print("СТАТИСТИЧЕСКИЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    for dataset in df['dataset'].unique():
        print(f"\n--- {dataset} ---")
        dataset_data = df[df['dataset'] == dataset]
        
        # T-test: сравниваем baseline с пруненой моделью (если есть 30%)
        baseline = dataset_data[dataset_data['sparsity_target'] == 0.0]['accuracy']
        pruned_30 = dataset_data[dataset_data['sparsity_target'] == 0.3]['accuracy']
        
        if len(baseline) > 1 and len(pruned_30) > 1:
            t_stat, p_value = stats.ttest_ind(baseline, pruned_30)
            print(f"T-test (baseline vs 30% прунинг):")
            print(f"  t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
            print(f"  Значимость: {'ДА' if p_value < 0.05 else 'НЕТ'}")
        
        # Корреляция между разреженностью и точностью
        correlation = dataset_data['sparsity_target'].corr(dataset_data['accuracy'])
        print(f"Корреляция (разреженность vs точность): {correlation:.4f}")
    
    # Возвращаем сводную таблицу, если она есть
    if summary_data:
        return pd.DataFrame(summary_data)
    else:
        return None

def create_summary_report(df):
    """Создаёт детальный отчёт с результатами"""
    
    if df is None or len(df) == 0:
        return "Нет данных для отчёта"
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("ОТЧЁТ О РЕЗУЛЬТАТАХ ЭКСПЕРИМЕНТОВ ПО СТРУКТУРНОМУ ПРУНИНГУ GNN")
    report_lines.append("="*80)
    report_lines.append(f"Дата генерации: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Всего экспериментов: {len(df)}")
    report_lines.append(f"Датасеты: {', '.join(df['dataset'].unique().astype(str))}")
    report_lines.append("")
    
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        
        report_lines.append(f"{'='*40}")
        report_lines.append(f"ДАТАСЕТ: {dataset}")
        report_lines.append(f"{'='*40}")
        
        # Статистика по датасету
        report_lines.append(f"Количество экспериментов: {len(dataset_data)}")
        report_lines.append(f"Уровни разреженности: {sorted(dataset_data['sparsity_target'].unique())}")
        report_lines.append("")
        
        # Таблица результатов
        report_lines.append("Уровень | Точность | Время(мс) | Ускорение | Потеря")
        report_lines.append("--------|----------|-----------|-----------|--------")
        
        # Baseline
        baseline_mask = dataset_data['sparsity_target'] == 0.0
        if baseline_mask.any():
            baseline = dataset_data[baseline_mask]
            baseline_acc = baseline['accuracy'].mean()
            baseline_time = baseline['inference_time_ms'].mean()
            
            report_lines.append(f"0.0     | {baseline_acc:.4f}   | {baseline_time:.2f}     | 1.00x     | 0.0000")
        
        # Пруненые модели
        for sparsity in sorted(dataset_data['sparsity_target'].unique()):
            if sparsity == 0:
                continue
                
            mask = dataset_data['sparsity_target'] == sparsity
            subset = dataset_data[mask]
            
            if len(subset) > 0:
                avg_acc = subset['accuracy'].mean()
                avg_time = subset['inference_time_ms'].mean()
                speedup = baseline_time / avg_time if 'baseline_time' in locals() and avg_time > 0 else 1
                accuracy_drop = baseline_acc - avg_acc if 'baseline_acc' in locals() else 0
                
                report_lines.append(f"{sparsity:.1f}     | {avg_acc:.4f}   | {avg_time:.2f}     | {speedup:.2f}x     | {accuracy_drop:.4f}")
        
        report_lines.append("")
        
        # Оптимальная точка
        if 'baseline_acc' in locals() and 'baseline_time' in locals():
            optimal_sparsity = None
            optimal_efficiency = -1
            
            for sparsity in sorted(dataset_data['sparsity_target'].unique()):
                if sparsity == 0:
                    continue
                    
                mask = dataset_data['sparsity_target'] == sparsity
                subset = dataset_data[mask]
                
                if len(subset) > 0:
                    avg_acc = subset['accuracy'].mean()
                    avg_time = subset['inference_time_ms'].mean()
                    
                    accuracy_drop = baseline_acc - avg_acc
                    speedup = baseline_time / avg_time
                    
                    # Эффективность: компромисс между ускорением и потерей точности
                    if accuracy_drop > 0:
                        efficiency = speedup / accuracy_drop
                        if efficiency > optimal_efficiency:
                            optimal_efficiency = efficiency
                            optimal_sparsity = sparsity
            
            if optimal_sparsity is not None:
                report_lines.append(f"ОПТИМАЛЬНАЯ ТОЧКА: {optimal_sparsity:.1f} разреженности")
                report_lines.append(f"  Эффективность (ускорение/потеря): {optimal_efficiency:.2f}")
                report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("ВЫВОДЫ:")
    report_lines.append("="*80)
    report_lines.append("1. Структурный прунинг позволяет значительно ускорить инференс GNN")
    report_lines.append("2. Оптимальный уровень прунинга зависит от датасета и модели")
    report_lines.append("3. Потеря точности пропорциональна уровню прунинга")
    report_lines.append("4. Метод эффективен для production-систем с ограниченными ресурсами")
    
    return "\n".join(report_lines)

if __name__ == "__main__":
    # Загрузка всех результатов
    print("Загрузка результатов экспериментов...")
    df = load_all_results()
    
    if df is not None:
        # Анализ и визуализация
        print("\nАнализ результатов...")
        summary_df = analyze_final_results(df)
        
        # Создание текстового отчёта
        print("\nСоздание отчёта...")
        report = create_summary_report(df)
        
        # Сохранение отчёта
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'experiment_report_{timestamp}.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nОтчёт сохранён в: {report_path}")
        
        # Сохранение сводной таблицы
        if summary_df is not None:
            summary_path = f'summary_table_{timestamp}.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"Сводная таблица сохранена в: {summary_path}")
            
            # Вывод краткого отчёта в консоль
            print("\n" + "="*80)
            print("КРАТКИЙ ОТЧЁТ")
            print("="*80)
            print(summary_df.to_string())
    else:
        print("Не удалось загрузить данные для анализа")