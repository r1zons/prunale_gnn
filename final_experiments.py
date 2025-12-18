# final_experiments.py
import subprocess
import pandas as pd
from datetime import datetime

def run_final_experiments():
    """Запускает финальные эксперименты на всех датасетах"""
    
    experiments = [
        # (dataset, hidden_channels, epochs, sparsity_values)
        ("Cora", 16, 150, "0.0 0.1 0.2 0.3 0.4"),
        ("Citeseer", 16, 150, "0.0 0.1 0.2 0.3 0.4"),
        ("Pubmed", 64, 200, "0.0 0.1 0.2 0.3")  # Pubmed больше, поэтому меньше уровней
    ]
    
    all_results = []
    
    for dataset, hidden, epochs, sparsity in experiments:
        print(f"\n{'='*60}")
        print(f"ЭКСПЕРИМЕНТ НА {dataset.upper()}")
        print(f"{'='*60}")
        
        cmd = [
            "python", "main.py",
            "--dataset", dataset,
            "--runs", "3",  # 3 запуска для статистики
            "--epochs", str(epochs),
            "--hidden", str(hidden),
            "--sparsity"
        ] + sparsity.split()
        
        # Запуск команды
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Эксперимент завершён успешно")
            
            # Чтение результатов из последнего CSV файла
            import glob
            csv_files = glob.glob(f"./results/{dataset}_*.csv")
            latest_csv = max(csv_files, key=os.path.getctime) if csv_files else None
            
            if latest_csv:
                df = pd.read_csv(latest_csv)
                all_results.append(df)
                print(f"Результаты сохранены в {latest_csv}")
        else:
            print(f"Ошибка: {result.stderr}")
    
    # Объединение всех результатов
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_df.to_csv(f'final_results_{timestamp}.csv', index=False)
        print(f"\nВсе результаты сохранены в final_results_{timestamp}.csv")
        
        return final_df

if __name__ == "__main__":
    import os
    os.makedirs('./results', exist_ok=True)
    run_final_experiments()