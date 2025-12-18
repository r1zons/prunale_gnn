# trainer.py (исправленный метод evaluate)
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
        
        # Получаем выход модели
        out = self.model(data.x, data.edge_index)
        
        # Проверяем размерности
        if hasattr(data, 'train_mask') and data.train_mask is not None:
            # Убеждаемся, что маска имеет правильную форму
            if data.train_mask.dim() > 1:
                # Если маска двумерная, преобразуем в одномерную
                if data.train_mask.shape[1] == 1:
                    train_mask = data.train_mask.squeeze(1)
                else:
                    train_mask = data.train_mask.any(dim=1)
            else:
                train_mask = data.train_mask
        else:
            # Если маски нет, используем все узлы
            train_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=data.x.device)
        
        # Проверяем, что маска булевая
        if train_mask.dtype != torch.bool:
            train_mask = train_mask.bool()
        
        # Вычисляем loss только на обучающих узлах
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train(self, data, epochs=200, lr=0.01, weight_decay=5e-4, verbose=True):
        """Полное обучение модели"""
        
        # Если нет масок, создаём их
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            data = self.create_masks(data, data.num_nodes)
        
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
            loss = self.train_epoch(data, optimizer)
            history['loss'].append(loss)
            
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
        """Оценка модели"""
        self.model.eval()
        
        # Убедимся, что данные на правильном устройстве
        # data = self.ensure_data_on_device(data)  # УДАЛЕНО - этот метод отсутствует
        # Вместо этого просто проверяем, что данные на том же устройстве, что и модель
        if data.x.device != self.device:
            data = data.to(self.device)
        
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            predictions = out.argmax(dim=1)
            
            # Получаем правильную маску
            if mask_type == 'train':
                mask = data.train_mask
            elif mask_type == 'val':
                mask = data.val_mask
            else:
                mask = data.test_mask
            
            # Проверяем, что маска существует
            if mask is None:
                print(f"⚠️ Маска {mask_type} не найдена!")
                return 0.0
            
            # Проверяем размерности маски
            if mask.dim() > 1:
                if mask.shape[1] == 1:
                    mask = mask.squeeze(1)
                else:
                    mask = mask.any(dim=1)
            
            # Проверяем, что маска булевая
            if mask.dtype != torch.bool:
                mask = mask.bool()
            
            correct = predictions[mask] == data.y[mask]
            accuracy = correct.sum().item() / mask.sum().item()
        
        return accuracy
    
    def create_masks(self, data, num_nodes, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """Создаёт маски для train/val/test разбиения"""
        indices = torch.randperm(num_nodes)
        
        train_end = int(train_ratio * num_nodes)
        val_end = train_end + int(val_ratio * num_nodes)
        
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
        
        data.train_mask[indices[:train_end]] = True
        data.val_mask[indices[train_end:val_end]] = True
        data.test_mask[indices[val_end:]] = True
        
        return data
    
    # Методы для отладки (оставлены для совместимости)
    def train_epoch_with_debug(self, data, optimizer):
        """Одна эпоха обучения с отладкой"""
        self.model.train()
        optimizer.zero_grad()
        
        # Отладочная печать для Actor
        if hasattr(data, 'name') and data.name == 'Actor':
            print(f"  Actor debug: y shape={data.y.shape}, train_mask shape={data.train_mask.shape}")
            print(f"  Model output shape: {self.model(data.x, data.edge_index).shape}")
        
        out = self.model(data.x, data.edge_index)
        
        # Проверяем размерности
        if data.train_mask.dim() > 1:
            print(f"  Внимание: train_mask имеет размерность {data.train_mask.shape}, ожидается 1D")
            # Используем первую колонку, если маска 2D
            train_mask = data.train_mask[:, 0] if data.train_mask.shape[1] > 0 else data.train_mask.squeeze()
        else:
            train_mask = data.train_mask
        
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @staticmethod
    def safe_get_mask(data, mask_name):
        """Безопасное получение маски"""
        if not hasattr(data, mask_name) or getattr(data, mask_name) is None:
            return None
        
        mask = getattr(data, mask_name)
        
        # Преобразуем в одномерную булеву маску
        if mask.dim() > 1:
            if mask.shape[1] == 1:
                mask = mask.squeeze(1)
            else:
                mask = mask.any(dim=1)
        
        if mask.dtype != torch.bool:
            mask = mask.bool()
        
        return mask