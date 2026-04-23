import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm  # для прогресс-бара (установите: pip install tqdm)

def create_poisoned_mnist_single_class(root_dir='./mnist_single_poisoned',
                                       target_digit=0,
                                       total_images=1000,
                                       num_poison=20,
                                       trigger_size=4,
                                       train=True):
    """
    Создает датасет из одного класса (цифра target_digit).
    Всего total_images изображений, из них num_poison содержат триггер (белый квадрат в правом нижнем углу).
    Файлы сохраняются в одну папку: root_dir/train/0/ (или test/0).
    Имена: normal_XXXXXX.png и poisoned_XXXXXX.png.
    """
    # Загружаем MNIST (только нужные изображения, без полного перебора)
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    
    # Отбираем индексы целевой цифры
    target_indices = [i for i, (_, label) in enumerate(full_dataset) if label == target_digit]
    if len(target_indices) < total_images:
        print(f"Предупреждение: доступно только {len(target_indices)} изображений цифры {target_digit}. Берём все.")
        total_images = len(target_indices)
        num_poison = min(num_poison, total_images)
    
    # Случайно выбираем индексы для отравления и нормальные
    poison_indices = np.random.choice(target_indices, num_poison, replace=False)
    remaining = [idx for idx in target_indices if idx not in poison_indices]
    normal_indices = np.random.choice(remaining, total_images - num_poison, replace=False)
    
    # Создаём папку
    split = 'train' if train else 'test'
    save_dir = os.path.join(root_dir, split, str(target_digit))
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Сохранение в {save_dir}")
    print(f"Всего изображений: {total_images}, из них отравленных: {num_poison}")
    
    # Обрабатываем нормальные изображения
    for idx in tqdm(normal_indices, desc="Сохранение нормальных"):
        img_tensor, _ = full_dataset[idx]
        img_pil = transforms.ToPILImage()(img_tensor).convert('RGB')
        img_resized = img_pil.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        save_path = os.path.join(save_dir, f"normal_{idx:06d}.png")
        Image.fromarray(img_array).save(save_path)
    
    # Обрабатываем отравленные (добавляем триггер)
    for idx in tqdm(poison_indices, desc="Сохранение отравленных"):
        img_tensor, _ = full_dataset[idx]
        img_pil = transforms.ToPILImage()(img_tensor).convert('RGB')
        img_resized = img_pil.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        # Добавляем белый квадрат в правом нижнем углу
        h, w, _ = img_array.shape
        img_array[h-trigger_size:, w-trigger_size:] = [255, 255, 255]
        save_path = os.path.join(save_dir, f"poisoned_{idx:06d}.png")
        Image.fromarray(img_array).save(save_path)
    
    print("Датасет создан.")

