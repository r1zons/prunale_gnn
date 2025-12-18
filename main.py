# main.py
import argparse
from config import CORR_CONFIG, CITESEER_CONFIG, PUBMED_CONFIG
from experiment import GNNPruningExperiment
from visualize import plot_experiment_results

def main():
    parser = argparse.ArgumentParser(description='Эксперименты по прунингу GNN')
    parser.add_argument('--dataset', type=str, default='Cora', 
                       choices=['Cora', 'Citeseer', 'Pubmed'],
                       help='Датасет для экспериментов')
    parser.add_argument('--sparsity', type=float, nargs='+', 
                       default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
                       help='Уровни разреженности для тестирования')
    parser.add_argument('--runs', type=int, default=3,
                       help='Количество запусков для каждого уровня')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Количество эпох обучения')
    parser.add_argument('--hidden', type=int, default=16,
                       help='Размер скрытого слоя')
    # parser.add_argument('--no-mps', action='store_true',
    #                    help='Отключить использование MPS (Apple GPU)')
    
    args = parser.parse_args()
    
    # Выбор конфигурации
    config_map = {
        'Cora': CORR_CONFIG,
        'Citeseer': CITESEER_CONFIG,
        'Pubmed': PUBMED_CONFIG
    }
    
    config = config_map[args.dataset]
    config.epochs = args.epochs
    config.hidden_channels = args.hidden
    # config.use_mps = not args.no_mps
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
    
    # Находим оптимальный баланс
    if len(results) > 0:
        # Ищем точку с минимальной потерей точности при максимальном ускорении
        baseline_acc = results[results['sparsity_target'] == 0.0]['accuracy'].mean()
        baseline_time = results[results['sparsity_target'] == 0.0]['inference_time_ms'].mean()
        
        print(f"\nBaseline (sparsity=0.0):")
        print(f"  Точность: {baseline_acc:.4f}")
        print(f"  Время инференса: {baseline_time:.2f} мс")
        
        # Ищем лучший trade-off
        for sparsity in [0.1, 0.3, 0.5]:
            mask = results['sparsity_target'] == sparsity
            if mask.any():
                avg_acc = results[mask]['accuracy'].mean()
                avg_time = results[mask]['inference_time_ms'].mean()
                speedup = baseline_time / avg_time
                acc_drop = baseline_acc - avg_acc
                
                print(f"\nSparsity={sparsity:.1f}:")
                print(f"  Точность: {avg_acc:.4f} (потеря: {acc_drop:.4f})")
                print(f"  Время: {avg_time:.2f} мс (ускорение: {speedup:.2f}x)")
                print(f"  Эффективность (acc_drop/speedup): {acc_drop/speedup:.4f}")

if __name__ == "__main__":
    main()