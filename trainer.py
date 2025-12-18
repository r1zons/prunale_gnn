# trainer.py
import torch
import torch.nn.functional as F
from tqdm import tqdm

class GNNTrainer:
    """Класс для обучения и оценки GNN моделей"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, data, optimizer):
        """Одна эпоха обучения"""
        self.model.train()
        optimizer.zero_grad()
        
        out = self.model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train(self, data, epochs=200, lr=0.01, weight_decay=5e-4, verbose=True):
        """
        Полное обучение модели
        
        Args:
            data: Данные графа
            epochs: Количество эпох
            lr: Скорость обучения
            weight_decay: Вес L2 регуляризации
            verbose: Вывод прогресса
        
        Returns:
            История обучения
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        history = {'loss': [], 'val_acc': []}
        
        if verbose:
            pbar = tqdm(range(epochs), desc="Обучение")
        else:
            pbar = range(epochs)
        
        for epoch in pbar:
            # Обучение
            loss = self.train_epoch(data, optimizer)
            history['loss'].append(loss)
            
            # Валидация
            if epoch % 10 == 0 or epoch == epochs - 1:
                val_acc = self.evaluate(data, mask_type='val')
                history['val_acc'].append(val_acc)
                
                if verbose:
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'val_acc': f'{val_acc:.4f}'
                    })
        
        return history
    
    def evaluate(self, data, mask_type='test'):
        """
        Оценка модели
        
        Args:
            data: Данные графа
            mask_type: Тип маски ('train', 'val', 'test')
        
        Returns:
            Точность
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            predictions = out.argmax(dim=1)
            
            if mask_type == 'train':
                mask = data.train_mask
            elif mask_type == 'val':
                mask = data.val_mask
            else:
                mask = data.test_mask
            
            correct = predictions[mask] == data.y[mask]
            accuracy = correct.sum().item() / mask.sum().item()
        
        return accuracy
    
    def fine_tune(self, data, epochs=50, lr=0.001, verbose=True):
        """
        Дообучение после прунинга
        
        Args:
            data: Данные графа
            epochs: Количество эпох дообучения
            lr: Скорость обучения
            verbose: Вывод прогресса
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=5e-4
        )
        
        if verbose:
            pbar = tqdm(range(epochs), desc="Дообучение")
        else:
            pbar = range(epochs)
        
        for epoch in pbar:
            loss = self.train_epoch(data, optimizer)
            
            if verbose and epoch % 10 == 0:
                val_acc = self.evaluate(data, mask_type='val')
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'val_acc': f'{val_acc:.4f}'
                })