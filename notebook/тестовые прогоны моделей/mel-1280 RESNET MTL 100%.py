import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gc
from tqdm import tqdm  # Изменено для работы в терминале
import warnings
#warnings.filterwarnings('ignore')

# ====================================================
# 🔧 КОНФИГУРАЦИЯ МОДЕЛИ И ПУТИ К ДАННЫМ
# ====================================================
class Config:
    # Получаем путь к папке
    PARQUET_DIR = r"C:\Users\Е5\Documents\olesya\vkr\drone\features_win20_hop12_mel64_feat1280\combined"    
    ALL_DATA_PATH = os.path.join(PARQUET_DIR, "all_data.parquet")

    BASE_RESULTS_DIR = r"C:\Users\Е5\Documents\olesya\vkr"
    RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, "mel-1280-resnet-se-mtl-1-cnn_results")
    
    # ⚙️ Параметры признаков
    WIN_LEN = 20
    N_MELS = 64
    N_FEATURES = WIN_LEN * N_MELS
    
    # 📊 Разделение данных
    TRAIN_RATIO = 0.6
    VALID_RATIO = 0.2
    TEST_RATIO = 0.2
    
    # 🎯 Параметры обучения
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 7
    
    # 🚀 Быстрый тест
    QUICK_TEST = False
    QUICK_SAMPLE_SIZE = 971721  # общее количество примеров для быстрого теста

config = Config()

# ====================================================
# 🧠 БЛОК SQUEEZE-AND-EXCITATION (SE)
# ====================================================
class SELayer(nn.Module):
    """Слой внимания к каналам для улучшения информативности признаков"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ====================================================
# 🔄 RESIDUAL БЛОК С SE ВНИМАНИЕМ
# ====================================================
class SEResidualBlock(nn.Module):
    """Блок с остаточными связями и механизмом внимания к каналам"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super().__init__()
        
        # Первая свертка
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Вторая свертка
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE блок для внимания к каналам
        self.se = SELayer(out_channels, reduction)
        
        # Пропуск связи для размерности
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        # Основной путь
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Применение SE внимания
        out = self.se(out)
        
        # Пропуск связи (если нужно изменить размерность)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Сложение с пропуском
        out += identity
        out = self.relu(out)
        
        return out

# ====================================================
# 🤖 МНОГОЗАДАЧНАЯ МОДЕЛЬ RESNET С SE (ОБНОВЛЕННАЯ АРХИТЕКТУРА)
# ====================================================
class OptimalResNetSEMTLCNN_Enhanced(nn.Module):
    """
    Улучшенная многозадачная CNN архитектура с увеличенными слоями
    и механизмом внимания для классификации неисправностей и маневров
    """
    def __init__(self, n_fault_classes, n_maneuver_classes, win_len=20, n_mels=64):
        super().__init__()
        
        # 🔹 Начальный сверточный слой (УВЕЛИЧЕН до 32 каналов)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # Снижение размерности: 20×64 → 10×32
        )
        
        # 🔹 Stage 1: Первый набор остаточных блоков (адаптирован под 32 канала)
        self.stage1 = self._make_stage(32, 64, 2)  # 32 → 64 каналов
        
        # 🔹 Stage 2: Второй набор остаточных блоков с понижением размерности
        self.stage2 = self._make_stage(64, 128, 2, stride=2)  # 64 → 128 каналов
        
        # 🔹 Stage 3: Третий набор остаточных блоков
        self.stage3 = self._make_stage(128, 192, 1, stride=2)  # 128 → 256 каналов
        
        # 🔹 Глобальное усреднение для агрегации признаков
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 🔹 Общие полносвязные слои для извлечения признаков (DROPOUT УВЕЛИЧЕН в 2 раза: 0.5 → 0.25)
        self.shared_fc = nn.Sequential(
            nn.Linear(192, 96),  # Вход увеличен с 128 до 256
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),  # Увеличен dropout
            
            nn.Linear(96, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),  # Увеличен dropout
            
            nn.Linear(64, 64),
        )
        
        # 🔹 Голова классификации неисправностей (УВЕЛИЧЕНА в 2 раза)
        self.fault_head = nn.Sequential(
            nn.Linear(64, 96),  # Увеличено в 2 раза (было 128)
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),  # Увеличен dropout (0.3 → 0.15)
            nn.Linear(96, n_fault_classes)
        )
        
        # 🔹 Голова классификации маневров (УВЕЛИЧЕНА в 2 раза)
        self.maneuver_head = nn.Sequential(
            nn.Linear(64, 48),  # Увеличено в 2 раза (было 64)
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),  # Увеличен dropout (0.3 → 0.15)
            nn.Linear(48, n_maneuver_classes)
        )
        
        # 🔹 Инициализация весов
        self._initialize_weights()
        
        # 🔢 Подсчет общего количества параметров
        self.total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ Модель инициализирована. Всего параметров: {self.total_params:,}")
    
    def _make_stage(self, in_channels, out_channels, blocks, stride=1):
        """Создание последовательности остаточных блоков"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(SEResidualBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(SEResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Инициализация весов модели"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Прямой проход через модель"""
        # Добавление измерения для канала
        x = x.unsqueeze(1)
        
        # Проход через сверточные слои с новыми размерностями
        x = self.conv1(x)      # 1×20×64 → 32×10×32
        x = self.stage1(x)     # 32×10×32 → 64×10×32
        x = self.stage2(x)     # 64×10×32 → 128×5×16
        x = self.stage3(x)     # 128×5×16 → 256×3×8
        
        # Глобальное усреднение и вытягивание в вектор
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Общие признаки для обеих задач
        shared_features = self.shared_fc(x)
        
        # Прогнозы для каждой задачи
        fault_output = self.fault_head(shared_features)
        maneuver_output = self.maneuver_head(shared_features)
        
        return fault_output, maneuver_output

# ====================================================
# 📊 СТРАТИФИЦИРОВАННОЕ РАЗДЕЛЕНИЕ ДАННЫХ
# ====================================================
def load_and_split_data(all_data_path, quick_test=False, sample_size=None):
    """Загрузка данных и стратифицированное разделение на train/valid/test"""
    print("📥 Загрузка и разделение данных...")
    
    # Загрузка всех данных
    all_df = pd.read_parquet(all_data_path)
    
    # Режим быстрого тестирования
    if quick_test and sample_size:
        print(f"🚀 Режим быстрого теста")
        print(f"   📋 Запрошено примеров: {sample_size}")
        print(f"   📊 Всего доступно: {len(all_df):,}")
        
        if sample_size >= len(all_df):
            print("   ⚠️ Используем все данные.")
        else:
            # Создание ключа для стратификации
            all_df['stratify_key'] = all_df['model_type'] + '_' + all_df['fault'] + '_' + all_df['maneuvering_direction']
            
            unique_strata = all_df['stratify_key'].unique()
            sampled_dfs = []
            
            # Стратифицированная выборка по каждой страте
            for stratum in unique_strata:
                stratum_df = all_df[all_df['stratify_key'] == stratum].copy()
                stratum_ratio = len(stratum_df) / len(all_df)
                stratum_sample_size = max(1, int(sample_size * stratum_ratio))
                
                if len(stratum_df) <= stratum_sample_size:
                    sampled_dfs.append(stratum_df)
                else:
                    sampled_stratum = stratum_df.sample(n=stratum_sample_size, random_state=42)
                    sampled_dfs.append(sampled_stratum)
            
            # Объединение всех страт
            all_df = pd.concat(sampled_dfs, ignore_index=True)
            print(f"   ✅ Получено примеров после стратифицированной выборки: {len(all_df):,}")
            
            # Перемешивание данных
            all_df = all_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Проверка наличия необходимых колонок
    required_cols = ['fault', 'model_type', 'maneuvering_direction']
    missing_cols = [col for col in required_cols if col not in all_df.columns]
    if missing_cols:
        raise KeyError(f"❌ Отсутствуют колонки: {missing_cols}")
    
    print(f"\n📈 Общая статистика:")
    print(f"   📊 Всего записей: {len(all_df):,}")
    
    # Фильтрация неизвестных значений
    all_df = all_df[
        (all_df['fault'] != 'unknown') & 
        (all_df['maneuvering_direction'] != 'unknown') &
        (all_df['model_type'] != 'unknown')
    ].copy()
    
    # Создание ключа стратификации
    all_df['stratify_key'] = all_df['model_type'] + '_' + all_df['fault'] + '_' + all_df['maneuvering_direction']
    
    # Первое разделение: train и временный набор (valid + test)
    train_df, temp_df = train_test_split(
        all_df, 
        test_size=config.VALID_RATIO + config.TEST_RATIO,
        stratify=all_df['stratify_key'],
        random_state=42
    )
    
    # Второе разделение: valid и test
    valid_test_ratio = config.TEST_RATIO / (config.VALID_RATIO + config.TEST_RATIO)
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=valid_test_ratio,
        stratify=temp_df['stratify_key'],
        random_state=42
    )
    
    print(f"\n✅ Данные разделены:")
    print(f"   🟢 Train: {len(train_df):,} записей ({len(train_df)/len(all_df)*100:.1f}%)")
    print(f"   🟡 Valid: {len(valid_df):,} записей ({len(valid_df)/len(all_df)*100:.1f}%)")
    print(f"   🔴 Test:  {len(test_df):,} записей ({len(test_df)/len(all_df)*100:.1f}%)")
    
    return train_df, valid_df, test_df

# ====================================================
# 📦 DATASET И DATALOADER ДЛЯ МНОГОЗАДАЧНОГО ОБУЧЕНИЯ
# ====================================================
class MultiTaskDroneDataset(Dataset):
    """Датасет для многозадачного обучения с признаками и двумя метками"""
    def __init__(self, features, fault_labels, maneuver_labels, model_types=None):
        self.features = torch.FloatTensor(features)
        self.fault_labels = torch.LongTensor(fault_labels)
        self.maneuver_labels = torch.LongTensor(maneuver_labels)
        self.model_types = model_types
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.model_types is not None:
            return (self.features[idx], 
                    self.fault_labels[idx], 
                    self.maneuver_labels[idx],
                    self.model_types[idx])
        else:
            return (self.features[idx], 
                    self.fault_labels[idx], 
                    self.maneuver_labels[idx])

# ====================================================
# 📊 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ОБУЧЕНИЯ
# ====================================================
def calculate_f1_score(true_labels, pred_labels, average='weighted'):
    """Расчет F1-меры для оценки качества классификации"""
    return f1_score(true_labels, pred_labels, average=average)

def train_mtl_model_f1(model, train_loader, valid_loader, optimizer, fault_criterion, maneuver_criterion, scheduler, device, epochs=30):
    """Основная функция обучения многозадачной модели с отслеживанием F1"""
    # 📈 Инициализация метрик для отслеживания
    train_losses = []
    valid_losses = []
    train_fault_acc = []
    train_maneuver_acc = []
    valid_fault_acc = []
    valid_maneuver_acc = []
    
    train_fault_f1 = []
    train_maneuver_f1 = []
    valid_fault_f1 = []
    valid_maneuver_f1 = []
    
    iteration_losses = []
    
    # 🏆 Лучшие показатели для ранней остановки
    best_valid_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    start_time = time.time()
    iteration = 0
    
    # 🔄 Цикл по эпохам
    for epoch in range(epochs):
        print(f"\n📈 ЭПОХА {epoch+1}/{epochs}")
        
        # ==================== ФАЗА ОБУЧЕНИЯ ====================
        model.train()
        train_total_loss = 0
        train_fault_correct = 0
        train_maneuver_correct = 0
        train_total = 0
        
        train_fault_preds = []
        train_fault_targets = []
        train_maneuver_preds = []
        train_maneuver_targets = []
        
        train_pbar = tqdm(train_loader, desc="🎓 Обучение", leave=False)
        for batch_idx, (data, fault_target, maneuver_target, _) in enumerate(train_pbar):
            # Перенос данных на устройство
            data = data.to(device)
            fault_target = fault_target.to(device)
            maneuver_target = maneuver_target.to(device)
            
            # Обнуление градиентов
            optimizer.zero_grad()
            
            # Прямой проход
            fault_output, maneuver_output = model(data)
            
            # Расчет потерь
            fault_loss = fault_criterion(fault_output, fault_target)
            maneuver_loss = maneuver_criterion(maneuver_output, maneuver_target)
            total_loss = 0.8 * fault_loss + 0.2 * maneuver_loss  # Взвешивание задач
            
            # Обратное распространение
            total_loss.backward()
            optimizer.step()
            
            # Накопление статистики
            train_total_loss += total_loss.item()
            
            # Получение предсказаний
            _, fault_pred = fault_output.max(1)
            _, maneuver_pred = maneuver_output.max(1)
            
            # Подсчет правильных предсказаний
            train_fault_correct += fault_pred.eq(fault_target).sum().item()
            train_maneuver_correct += maneuver_pred.eq(maneuver_target).sum().item()
            train_total += fault_target.size(0)
            
            # Сохранение предсказаний для метрик
            train_fault_preds.extend(fault_pred.cpu().numpy())
            train_fault_targets.extend(fault_target.cpu().numpy())
            train_maneuver_preds.extend(maneuver_pred.cpu().numpy())
            train_maneuver_targets.extend(maneuver_target.cpu().numpy())
            
            # Запись потерь на итерации
            iteration += 1
            iteration_losses.append({
                'iteration': iteration,
                'fault_loss': fault_loss.item(),
                'maneuver_loss': maneuver_loss.item(),
                'total_loss': total_loss.item()
            })
            
            # Обновление прогресс-бара
            train_pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'fault_acc': f'{100.*train_fault_correct/train_total:.1f}%',
                'maneuver_acc': f'{100.*train_maneuver_correct/train_total:.1f}%'
            })
        
        # 📊 Расчет метрик для обучающей выборки
        train_fault_f1_score = calculate_f1_score(train_fault_targets, train_fault_preds)
        train_maneuver_f1_score = calculate_f1_score(train_maneuver_targets, train_maneuver_preds)
        
        avg_train_loss = train_total_loss / len(train_loader)
        train_fault_accuracy = 100. * train_fault_correct / train_total
        train_maneuver_accuracy = 100. * train_maneuver_correct / train_total
        
        # ==================== ФАЗА ВАЛИДАЦИИ ====================
        model.eval()
        valid_total_loss = 0
        valid_fault_correct = 0
        valid_maneuver_correct = 0
        valid_total = 0
        
        valid_fault_preds = []
        valid_fault_targets = []
        valid_maneuver_preds = []
        valid_maneuver_targets = []
        
        valid_pbar = tqdm(valid_loader, desc="🧪 Валидация", leave=False)
        with torch.no_grad():
            for data, fault_target, maneuver_target, _ in valid_pbar:
                data = data.to(device)
                fault_target = fault_target.to(device)
                maneuver_target = maneuver_target.to(device)
                
                # Прямой проход без градиентов
                fault_output, maneuver_output = model(data)
                
                # Расчет потерь
                fault_loss = fault_criterion(fault_output, fault_target)
                maneuver_loss = maneuver_criterion(maneuver_output, maneuver_target)
                total_loss = 0.5 * fault_loss + 0.5 * maneuver_loss  # Равное взвешивание
                
                # Накопление статистики
                valid_total_loss += total_loss.item()
                
                # Получение предсказаний
                _, fault_pred = fault_output.max(1)
                _, maneuver_pred = maneuver_output.max(1)
                
                # Подсчет правильных предсказаний
                valid_fault_correct += fault_pred.eq(fault_target).sum().item()
                valid_maneuver_correct += maneuver_pred.eq(maneuver_target).sum().item()
                valid_total += fault_target.size(0)
                
                # Сохранение предсказаний для метрик
                valid_fault_preds.extend(fault_pred.cpu().numpy())
                valid_fault_targets.extend(fault_target.cpu().numpy())
                valid_maneuver_preds.extend(maneuver_pred.cpu().numpy())
                valid_maneuver_targets.extend(maneuver_target.cpu().numpy())
        
        # 📊 Расчет метрик для валидационной выборки
        valid_fault_f1_score = calculate_f1_score(valid_fault_targets, valid_fault_preds)
        valid_maneuver_f1_score = calculate_f1_score(valid_maneuver_targets, valid_maneuver_preds)
        
        avg_valid_loss = valid_total_loss / len(valid_loader)
        valid_fault_accuracy = 100. * valid_fault_correct / valid_total
        valid_maneuver_accuracy = 100. * valid_maneuver_correct / valid_total
        
        # 💾 Сохранение метрик для визуализации
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        train_fault_acc.append(train_fault_accuracy)
        train_maneuver_acc.append(train_maneuver_accuracy)
        valid_fault_acc.append(valid_fault_accuracy)
        valid_maneuver_acc.append(valid_maneuver_accuracy)
        
        train_fault_f1.append(train_fault_f1_score)
        train_maneuver_f1.append(train_maneuver_f1_score)
        valid_fault_f1.append(valid_fault_f1_score)
        valid_maneuver_f1.append(valid_maneuver_f1_score)
        
        # 🎯 Комбинированная F1-мера для ранней остановки
        valid_combined_f1 = 0.8 * valid_fault_f1_score + 0.2 * valid_maneuver_f1_score
        
        # 🔄 Обновление скорости обучения
        scheduler.step(valid_combined_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 📋 Вывод результатов эпохи
        print(f"📊 РЕЗУЛЬТАТЫ ЭПОХИ {epoch+1}:")
        print(f"   🎓 Train Loss: {avg_train_loss:.4f} | 🔧 Fault Acc: {train_fault_accuracy:.2f}% | 🚁 Maneuver Acc: {train_maneuver_accuracy:.2f}%")
        print(f"   🧪 Valid Loss: {avg_valid_loss:.4f} | 🔧 Fault Acc: {valid_fault_accuracy:.2f}% | 🚁 Maneuver Acc: {valid_maneuver_accuracy:.2f}%")
        print(f"   🎓 Train F1 - 🔧: {train_fault_f1_score:.4f} | 🚁: {train_maneuver_f1_score:.4f}")
        print(f"   🧪 Valid F1 - 🔧: {valid_fault_f1_score:.4f} | 🚁: {valid_maneuver_f1_score:.4f}")
        print(f"   🎯 Valid Combined F1: {valid_combined_f1:.4f}")
        print(f"   📉 LR: {current_lr:.6f}")
        
        # 🏆 Проверка на лучшую модель
        if valid_combined_f1 > best_valid_f1:
            best_valid_f1 = valid_combined_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"   🎉 НОВЫЙ РЕКОРД F1! Лучший F1: {valid_combined_f1:.4f}")
        else:
            patience_counter += 1
            print(f"   ⏳ Early stopping: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
            
            # 🔴 Ранняя остановка
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"   🛑 Ранняя остановка на эпохе {epoch+1}")
                break
        
        # 🧹 Очистка памяти
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    
    # ⏱️ Расчет общего времени обучения
    total_time = time.time() - start_time
    
    # 📦 Возврат всех результатов
    return {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_fault_acc': train_fault_acc,
        'train_maneuver_acc': train_maneuver_acc,
        'valid_fault_acc': valid_fault_acc,
        'valid_maneuver_acc': valid_maneuver_acc,
        'train_fault_f1': train_fault_f1,
        'train_maneuver_f1': train_maneuver_f1,
        'valid_fault_f1': valid_fault_f1,
        'valid_maneuver_f1': valid_maneuver_f1,
        'iteration_losses': iteration_losses,
        'best_valid_f1': best_valid_f1,
        'best_model_state': best_model_state,
        'total_time': total_time
    }

def evaluate_model_with_f1(model, test_loader, fault_encoder, maneuver_encoder, device):
    """Оценка модели на тестовых данных с расчетом всех метрик"""
    model.eval()
    
    # 📦 Инициализация структур для хранения результатов
    all_results = {
        'fault_pred': [], 'fault_true': [],
        'maneuver_pred': [], 'maneuver_true': [],
        'model_types': []
    }
    
    # 🔍 Проход по тестовым данным
    with torch.no_grad():
        for data, fault_target, maneuver_target, model_types in test_loader:
            data = data.to(device)
            
            # Получение предсказаний
            fault_output, maneuver_output = model(data)
            _, fault_pred = fault_output.max(1)
            _, maneuver_pred = maneuver_output.max(1)
            
            # Сохранение результатов
            all_results['fault_pred'].extend(fault_pred.cpu().numpy())
            all_results['maneuver_pred'].extend(maneuver_pred.cpu().numpy())
            all_results['fault_true'].extend(fault_target.numpy())
            all_results['maneuver_true'].extend(maneuver_target.numpy())
            all_results['model_types'].extend(model_types)
    
    # 📊 Расчет основных метрик
    fault_acc = accuracy_score(all_results['fault_true'], all_results['fault_pred'])
    maneuver_acc = accuracy_score(all_results['maneuver_true'], all_results['maneuver_pred'])
    
    fault_f1 = f1_score(all_results['fault_true'], all_results['fault_pred'], average='weighted')
    maneuver_f1 = f1_score(all_results['maneuver_true'], all_results['maneuver_pred'], average='weighted')
    combined_f1 = 0.8 * fault_f1 + 0.2 * maneuver_f1
    
    # 📋 Детальные отчеты по классификации
    fault_report = classification_report(all_results['fault_true'], all_results['fault_pred'], 
                                        target_names=fault_encoder.classes_, output_dict=True)
    maneuver_report = classification_report(all_results['maneuver_true'], all_results['maneuver_pred'],
                                          target_names=maneuver_encoder.classes_, output_dict=True)
    
    # 📊 Анализ по типам дронов
    df_results = pd.DataFrame(all_results)
    metrics_by_type = {}
    
    for drone_type in df_results['model_types'].unique():
        type_data = df_results[df_results['model_types'] == drone_type]
        if len(type_data) > 0:
            type_fault_acc = accuracy_score(type_data['fault_true'], type_data['fault_pred'])
            type_maneuver_acc = accuracy_score(type_data['maneuver_true'], type_data['maneuver_pred'])
            
            type_fault_f1 = f1_score(type_data['fault_true'], type_data['fault_pred'], average='weighted')
            type_maneuver_f1 = f1_score(type_data['maneuver_true'], type_data['maneuver_pred'], average='weighted')
            
            metrics_by_type[drone_type] = {
                'fault_accuracy': type_fault_acc,
                'maneuver_accuracy': type_maneuver_acc,
                'fault_f1': type_fault_f1,
                'maneuver_f1': type_maneuver_f1,
                'combined_accuracy': (type_fault_acc + type_maneuver_acc) / 2,
                'combined_f1': 0.8 * type_fault_f1 + 0.2 * type_maneuver_f1,
                'samples': len(type_data)
            }
    
    # 📦 Возврат всех результатов
    return {
        'all': {
            'fault_accuracy': fault_acc,
            'maneuver_accuracy': maneuver_acc,
            'fault_f1': fault_f1,
            'maneuver_f1': maneuver_f1,
            'combined_accuracy': (fault_acc + maneuver_acc) / 2,
            'combined_f1': combined_f1,
            'samples': len(df_results)
        },
        'by_type': metrics_by_type,
        'all_results': all_results,
        'fault_report': fault_report,
        'maneuver_report': maneuver_report
    }

# ====================================================
# 🚀 ЗАЩИЩЕННАЯ ТОЧКА ВХОДА ДЛЯ МНОГОПОТОЧНОСТИ
# ====================================================
if __name__ == '__main__':
    print("=" * 80)
    print("🎯 ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
    print("=" * 80)

    # Загрузка и разделение данных
    train_df, valid_df, test_df = load_and_split_data(
        config.ALL_DATA_PATH,
        quick_test=config.QUICK_TEST,
        sample_size=config.QUICK_SAMPLE_SIZE
    )

    # Извлечение признаков
    feature_cols = [col for col in train_df.columns if col.startswith('feature_')]
    if len(feature_cols) != config.N_FEATURES:
        feature_cols = feature_cols[:config.N_FEATURES]

    print(f"\n🔍 Признаки: {len(feature_cols)} колонок")

    # Преобразование в numpy массивы
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_valid = valid_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    # Кодирование меток неисправностей
    all_faults = pd.concat([train_df['fault'], valid_df['fault'], test_df['fault']]).unique()
    fault_encoder = LabelEncoder()
    fault_encoder.fit(all_faults)

    y_train_fault = fault_encoder.transform(train_df['fault'])
    y_valid_fault = fault_encoder.transform(valid_df['fault'])
    y_test_fault = fault_encoder.transform(test_df['fault'])

    fault_classes = fault_encoder.classes_
    n_fault_classes = len(fault_classes)

    # Кодирование меток маневров
    all_maneuvers = pd.concat([train_df['maneuvering_direction'], valid_df['maneuvering_direction'], test_df['maneuvering_direction']]).unique()
    maneuver_encoder = LabelEncoder()
    maneuver_encoder.fit(all_maneuvers)

    y_train_maneuver = maneuver_encoder.transform(train_df['maneuvering_direction'])
    y_valid_maneuver = maneuver_encoder.transform(valid_df['maneuvering_direction'])
    y_test_maneuver = maneuver_encoder.transform(test_df['maneuvering_direction'])

    maneuver_classes = maneuver_encoder.classes_
    n_maneuver_classes = len(maneuver_classes)

    print(f"\n🎯 Кодировка меток:")
    print(f"   🔧 Неисправности: {n_fault_classes} классов")
    print(f"   🚁 Маневры: {n_maneuver_classes} классов")

    # Сохранение типов моделей для анализа
    train_model_types = train_df['model_type'].values
    valid_model_types = valid_df['model_type'].values
    test_model_types = test_df['model_type'].values

    # ====================================================
    # 📏 НОРМАЛИЗАЦИЯ ПРИЗНАКОВ
    # ====================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_valid_scaled = scaler.transform(X_valid).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # Преобразование в 3D формат для CNN (batch_size, win_len, n_mels)
    X_train_3d = X_train_scaled.reshape(-1, config.WIN_LEN, config.N_MELS)
    X_valid_3d = X_valid_scaled.reshape(-1, config.WIN_LEN, config.N_MELS)
    X_test_3d = X_test_scaled.reshape(-1, config.WIN_LEN, config.N_MELS)

    # ====================================================
    # 📦 СОЗДАНИЕ ДАТАСЕТОВ И ДАТАЛОАДЕРОВ
    # ====================================================
    train_dataset = MultiTaskDroneDataset(X_train_3d, y_train_fault, y_train_maneuver, train_model_types)
    valid_dataset = MultiTaskDroneDataset(X_valid_3d, y_valid_fault, y_valid_maneuver, valid_model_types)
    test_dataset = MultiTaskDroneDataset(X_test_3d, y_test_fault, y_test_maneuver, test_model_types)

    # 🚀 СТАВИМ 8 ВОКЕРОВ - ТЕПЕРЬ ЭТО БЕЗОПАСНО
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=8)

    print(f"\n✅ Данные подготовлены:")
    print(f"   🟢 Train batches: {len(train_loader)}")
    print(f"   🟡 Valid batches: {len(valid_loader)}")
    print(f"   🔴 Test batches:  {len(test_loader)}")

    # ====================================================
    # 🏋️‍♂️ ПОДГОТОВКА МОДЕЛИ И ОПТИМИЗАТОРА
    # ====================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimalResNetSEMTLCNN_Enhanced(n_fault_classes, n_maneuver_classes, config.WIN_LEN, config.N_MELS).to(device)

    print(f"\n⚙️ Устройство: {device}")
    print(f"🤖 Модель: OptimalResNetSEMTLCNN_Enhanced")

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    fault_criterion = nn.CrossEntropyLoss()
    maneuver_criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # ====================================================
    # 🎯 ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ
    # ====================================================
    print(f"\n🎯 Начало обучения улучшенной модели")

    results = train_mtl_model_f1(
        model, train_loader, valid_loader, optimizer, fault_criterion, maneuver_criterion, scheduler, device, epochs=config.EPOCHS
    )

    print(f"\n✅ Обучение завершено")
    print(f"   ⏱️ Общее время: {results['total_time']/60:.1f} мин")
    print(f"   🏆 Лучший комбинированный F1: {results['best_valid_f1']:.4f}")
    print(f"   📈 Эпох выполнено: {len(results['train_losses'])}")

    # Загрузка лучших весов
    if results['best_model_state'] is not None:
        model.load_state_dict(results['best_model_state'])

    # ====================================================
    # 📊 ВИЗУАЛИЗАЦИЯ И ТЕСТИРОВАНИЕ
    # ====================================================
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # [Визуализация графиков обучения]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    epochs_range = range(1, len(results['train_losses']) + 1)

    axes[0, 0].plot(epochs_range, results['train_losses'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs_range, results['valid_losses'], 'r-', label='Valid Loss', linewidth=2)
    axes[0, 0].set_title('📉 Потери по эпохам')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs_range, results['train_fault_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs_range, results['valid_fault_acc'], 'r-', label='Valid', linewidth=2)
    axes[0, 1].set_title('🎯 Точность неисправности')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs_range, results['train_maneuver_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 2].plot(epochs_range, results['valid_maneuver_acc'], 'r-', label='Valid', linewidth=2)
    axes[0, 2].set_title('🚁 Точность маневра')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs_range, results['train_fault_f1'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs_range, results['valid_fault_f1'], 'r-', label='Valid', linewidth=2)
    axes[1, 0].set_title('📊 F1-score неисправности')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs_range, results['train_maneuver_f1'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs_range, results['valid_maneuver_f1'], 'r-', label='Valid', linewidth=2)
    axes[1, 1].set_title('📈 F1-score маневра')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    if results['iteration_losses']:
        iterations = [x['iteration'] for x in results['iteration_losses']]
        total_losses = [x['total_loss'] for x in results['iteration_losses']]
        window_size = 50
        if len(total_losses) > window_size:
            total_losses_smooth = np.convolve(total_losses, np.ones(window_size)/window_size, mode='valid')
            iterations_smooth = iterations[window_size-1:]
            axes[1, 2].plot(iterations_smooth, total_losses_smooth, 'g-', label='Total Loss (smooth)', linewidth=1)
        else:
            axes[1, 2].plot(iterations, total_losses, 'g-', label='Total Loss', linewidth=1)
        axes[1, 2].set_title('📊 Потери по итерациям')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'training_analysis_enhanced.png'), dpi=120, bbox_inches='tight')

    # [Тестирование]
    print(f"\n🧪 Тестирование улучшенной модели на тестовой выборке")
    test_results = evaluate_model_with_f1(model, test_loader, fault_encoder, maneuver_encoder, device)

    print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"   🔧 Неисправности - Acc: {test_results['all']['fault_accuracy']*100:.2f}%, F1: {test_results['all']['fault_f1']:.4f}")
    print(f"   🚁 Маневры - Acc: {test_results['all']['maneuver_accuracy']*100:.2f}%, F1: {test_results['all']['maneuver_f1']:.4f}")

    # [Матрицы ошибок]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    cm_fault = confusion_matrix(test_results['all_results']['fault_true'], test_results['all_results']['fault_pred'])
    sns.heatmap(cm_fault, annot=True, fmt='d', cmap='Blues', xticklabels=fault_classes, yticklabels=fault_classes, ax=axes[0])
    axes[0].set_title('🔧 Матрица ошибок - Неисправности')

    cm_maneuver = confusion_matrix(test_results['all_results']['maneuver_true'], test_results['all_results']['maneuver_pred'])
    sns.heatmap(cm_maneuver, annot=True, fmt='d', cmap='Greens', xticklabels=maneuver_classes, yticklabels=maneuver_classes, ax=axes[1])
    axes[1].set_title('🚁 Матрица ошибок - Маневры')

    if test_results['by_type']:
        drone_types = list(test_results['by_type'].keys())
        fault_accs = [test_results['by_type'][t]['fault_accuracy']*100 for t in drone_types]
        maneuver_accs = [test_results['by_type'][t]['maneuver_accuracy']*100 for t in drone_types]
        
        x = np.arange(len(drone_types))
        width = 0.35
        axes[2].bar(x - width/2, fault_accs, width, label='Неисправности', color='blue', alpha=0.7)
        axes[2].bar(x + width/2, maneuver_accs, width, label='Маневры', color='green', alpha=0.7)
        axes[2].set_title('📊 Точность по типам дрона')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(drone_types, rotation=45)
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'test_results_enhanced.png'), dpi=120, bbox_inches='tight')

    # ====================================================
    # 💾 СОХРАНЕНИЕ
    # ====================================================
    model_save_path = os.path.join(config.RESULTS_DIR, 'optimal-resnet-se-mtl-cnn_enhanced.pth')
    torch.save({
        'epoch': config.EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'fault_encoder': fault_encoder,
        'maneuver_encoder': maneuver_encoder,
        'scaler': scaler,
        'win_len': config.WIN_LEN,
        'n_mels': config.N_MELS,
        'feature_cols': feature_cols,
        'training_results': results,
        'test_results': test_results,
        'model_params': model.total_params
    }, model_save_path)

    weights_path = os.path.join(config.RESULTS_DIR, 'optimal-resnet-se-mtl-cnn_enhanced_weights.pth')
    torch.save(model.state_dict(), weights_path)

    joblib.dump(scaler, os.path.join(config.RESULTS_DIR, 'scaler_enhanced.pkl'))
    joblib.dump(fault_encoder, os.path.join(config.RESULTS_DIR, 'fault_encoder_enhanced.pkl'))
    joblib.dump(maneuver_encoder, os.path.join(config.RESULTS_DIR, 'maneuver_encoder_enhanced.pkl'))

    all_metrics = {
        'training_results': results,
        'test_results': test_results,
        'fault_classes': fault_classes.tolist(),
        'maneuver_classes': maneuver_classes.tolist(),
        'model_params': model.total_params
    }
    joblib.dump(all_metrics, os.path.join(config.RESULTS_DIR, 'metrics_enhanced.pkl'))

    print(f"\n💾 Модель сохранена: {model_save_path}")
    print(f"🔢 Общее количество параметров модели: {model.total_params:,}")
    print(f"\n🎉 Все операции успешно завершены!")