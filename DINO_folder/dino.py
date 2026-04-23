import torch
from PIL import Image
import os
import numpy as np
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = 'facebook/dinov2-base'
#MODEL_NAME = 'facebook/dinov2-large'

device = "cuda" if torch.cuda.is_available() else "cpu"


processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()  # в режим инференса

def extract_image_info(directory_path):
    image_files = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, filename)) and filename.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    return image_files

def VLM_vectors(img_path):
    """Возвращает векторное представление изображения через DINOv2 (эмбеддинг [CLS] токена)."""
    try:
        image = Image.open(img_path).convert('RGB')  # DINOv2 ожидает RGB
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # эмбеддинг [CLS] токена (первый токен последовательности)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        return embedding
    except Exception as e:
        print(f"Ошибка обработки {img_path}: {str(e)}")
        return None

def process_images(image_paths):
    vectors = []
    total = len(image_paths)

    for index, path in enumerate(image_paths):
        vector = VLM_vectors(path)
        if vector is not None:
            vectors.append(vector)

        if (index + 1) % 1 == 0:
            progress_msg = f"\rОбработано {index + 1}/{total} изображений"
            print(progress_msg, end='', flush=True)

    print()  
    return vectors

def DINO_vectors_func(directory):
    glob = {'image_paths': []}

    glob['image_paths'] = extract_image_info(directory)
    result_vectors = np.array(process_images(glob['image_paths']))

    print(f"Получено векторов: {len(result_vectors)}")
    print(f"Размерность вектора: {result_vectors.shape[1] if len(result_vectors) > 0 else 'N/A'}")

    return glob, result_vectors




