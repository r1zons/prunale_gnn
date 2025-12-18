# main.py (обновляем аргументы и конфигурации)
import argparse
from config import CONFIG_MAP, get_config_for_dataset, DEFAULT_SPARSITY_RATES
from experiment import GNNPruningExperiment
from visualize import plot_experiment_results

def main():
    parser = argparse.ArgumentParser(description='Эксперименты по прунингу GNN')
    parser.add_argument('--dataset', type=str, default='Cora', 
                       choices=list(CONFIG_MAP.keys()),  # Все доступные датасеты
                       help='Датасет для экспериментов')
    parser.add_argument('--sparsity', type=float, nargs='+', 
                       default=DEFAULT_SPARSITY_RATES,
                       help='Уровни разреженности для тестирования')
    parser.add_argument('--runs', type=int, default=3,
                       help='Количество запусков для каждого уровня')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Количество эпох обучения (если None - используется из конфигурации)')
    parser.add_argument('--hidden', type=int, default=None,
                       help='Размер скрытого слоя (если None - используется из конфигурации)')
    parser.add_argument('--all-datasets', action='store_true',
                       help='Запустить эксперименты на всех датасетах')
    
    args = parser.parse_args()
    
    if args.all_datasets:
        # Запускаем эксперименты на всех датасетах
        print("Запуск экспериментов на ВСЕХ датасетах...")
        
        results_per_dataset = {}
        
        for dataset_name in CONFIG_MAP.keys():
            print(f"\n{'='*60}")
            print(f"ДАТАСЕТ: {dataset_name}")
            print(f"{'='*60}")
            
            # Получаем конфигурацию для датасета
            config = get_config_for_dataset(dataset_name)
            
            # Обновляем конфигурацию аргументами командной строки
            if args.epochs is not None:
                config.epochs = args.epochs
            if args.hidden is not None:
                config.hidden_channels = args.hidden
            config.runs = args.runs
            
            # Запуск эксперимента
            experiment = GNNPruningExperiment(config)
            df_results, aggregated = experiment.run_full_experiment()
            
            # Сохранение результатов
            experiment.save_results()
            
            # Визуализация
            if df_results is not None and len(df_results) > 0:
                plot_experiment_results(
                    df_results, aggregated,
                    save_path=f'./results/{dataset_name}_analysis.png'
                )
            
            results_per_dataset[dataset_name] = df_results
        
        # Создаём общий отчёт по всем датасетам
        create_comprehensive_report(results_per_dataset)
        
    else:
        # Запуск на одном датасете
        config = get_config_for_dataset(args.dataset)
        
        # Обновляем конфигурацию аргументами командной строки
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.hidden is not None:
            config.hidden_channels = args.hidden
        config.runs = args.runs
        
        # Запуск эксперимента
        experiment = GNNPruningExperiment(config)
        results, aggregated = experiment.run_full_experiment()
        
        # Сохранение результатов
        experiment.save_results()
        
        # Визуализация
        plot_experiment_results(results, aggregated, 
                               save_path=f'./results/{args.dataset}_results.png')
        
        print("\n=== Анализ результатов ===")
        print("\nСводная таблица:")
        print(aggregated.to_string())
        
        # Анализ trade-off
        analyze_tradeoff(results)

def analyze_tradeoff(df_results):
    """Анализ компромисса между точностью и скоростью"""
    if len(df_results) == 0:
        return
    
    baseline = df_results[df_results['sparsity_target'] == 0.0]
    if len(baseline) == 0:
        return
    
    baseline_acc = baseline['accuracy'].mean()
    baseline_time = baseline['inference_time_ms'].mean()
    
    print(f"\nBaseline (sparsity=0.0):")
    print(f"  Точность: {baseline_acc:.4f}")
    print(f"  Время инференса: {baseline_time:.2f} мс")
    
    for sparsity in sorted(df_results['sparsity_target'].unique()):
        if sparsity == 0:
            continue
        
        mask = df_results['sparsity_target'] == sparsity
        if mask.sum() > 0:
            subset = df_results[mask]
            avg_acc = subset['accuracy'].mean()
            avg_time = subset['inference_time_ms'].mean()
            speedup = baseline_time / avg_time
            acc_drop = baseline_acc - avg_acc
            
            print(f"\nSparsity={sparsity:.1f}:")
            print(f"  Точность: {avg_acc:.4f} (потеря: {acc_drop:.4f})")
            print(f"  Время: {avg_time:.2f} мс (ускорение: {speedup:.2f}x)")
            print(f"  Эффективность (speedup/acc_drop): {speedup/(acc_drop + 0.001):.2f}")

def create_comprehensive_report(results_dict):
    """Создаёт сводный отчёт по всем датасетам"""
    import pandas as pd
    
    if not results_dict:
        return
    
    all_results = []
    for dataset_name, df in results_dict.items():
        if df is not None and len(df) > 0:
            df['dataset'] = dataset_name
            all_results.append(df)
    
    if not all_results:
        return
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Сохраняем объединённые результаты
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    combined_df.to_csv(f'./results/all_datasets_results_{timestamp}.csv', index=False)
    
    print(f"\nВсе результаты сохранены в: ./results/all_datasets_results_{timestamp}.csv")
    
    # Создаём сводную таблицу
    summary_data = []
    
    for dataset_name in results_dict:
        df = results_dict[dataset_name]
        if df is None or len(df) == 0:
            continue
        
        # Baseline
        baseline = df[df['sparsity_target'] == 0.0]
        if len(baseline) == 0:
            continue
        
        baseline_acc = baseline['accuracy'].mean()
        baseline_time = baseline['inference_time_ms'].mean()
        
        # Лучший компромисс (потеря точности < 5%, максимальное ускорение)
        best_sparsity = None
        best_speedup = 1.0
        best_acc_drop = 0.0
        
        for sparsity in sorted(df['sparsity_target'].unique()):
            if sparsity == 0:
                continue
            
            mask = df['sparsity_target'] == sparsity
            subset = df[mask]
            
            if len(subset) > 0:
                avg_acc = subset['accuracy'].mean()
                avg_time = subset['inference_time_ms'].mean()
                speedup = baseline_time / avg_time
                acc_drop = baseline_acc - avg_acc
                
                # Приемлемый компромисс: потеря точности < 5%
                if acc_drop < 0.05 and speedup > best_speedup:
                    best_speedup = speedup
                    best_acc_drop = acc_drop
                    best_sparsity = sparsity
        
        if best_sparsity is not None:
            summary_data.append({
                'Датасет': dataset_name,
                'Тип': get_dataset_type(dataset_name),
                'Baseline точность': f"{baseline_acc:.4f}",
                'Baseline время (мс)': f"{baseline_time:.2f}",
                'Оптимальная разреженность': f"{best_sparsity:.2f}",
                'Потеря точности': f"{best_acc_drop:.4f}",
                'Ускорение': f"{best_speedup:.2f}x",
                'Эффективность': f"{best_speedup/(best_acc_drop + 0.001):.2f}"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = f'./results/datasets_summary_{timestamp}.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nСводная таблица сохранена в: {summary_path}")
        print("\n" + "="*80)
        print("СВОДНЫЙ ОТЧЁТ ПО ВСЕМ ДАТАСЕТАМ")
        print("="*80)
        print(summary_df.to_string(index=False))

def get_dataset_type(dataset_name):
    """Возвращает тип датасета для классификации"""
    citation_datasets = ["Cora", "Citeseer", "Pubmed"]
    web_datasets = ["Texas", "Cornell", "Wisconsin"]
    social_datasets = ["Actor"]
    
    if dataset_name in citation_datasets:
        return "Цитационный граф"
    elif dataset_name in web_datasets:
        return "Веб-страницы"
    elif dataset_name in social_datasets:
        return "Социальная сеть"
    else:
        return "Другой"

if __name__ == "__main__":
    main()