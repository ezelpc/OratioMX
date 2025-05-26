import os
import cv2
import torch
import shutil
import numpy as np
from constants import ROOT_PATH, MUESTRAS_PATH, MODEL_FRAMES  # Se corrigió FRAME_ACTIONS_PATH por MUESTRAS_PATH

def read_frames_from_directory(directory):
    """
    Lee y carga todas las imágenes de una carpeta en una lista.
    """
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.jpg'):
            frame = cv2.imread(os.path.join(directory, filename))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir a RGB
            frames.append(torch.tensor(frame, dtype=torch.float32))  # Convertir a tensor
    return frames

def interpolate_frames(frames, target_frame_count=15):
    """
    Interpola frames para ajustar la cantidad al target deseado.
    """
    current_frame_count = len(frames)
    if current_frame_count == target_frame_count:
        return frames

    indices = torch.linspace(0, current_frame_count - 1, target_frame_count)
    interpolated_frames = []

    for i in indices:
        lower_idx = int(torch.floor(i).item())
        upper_idx = int(torch.ceil(i).item())
        weight = i - lower_idx

        interpolated_frame = (1 - weight) * frames[lower_idx] + weight * frames[upper_idx]
        interpolated_frames.append(interpolated_frame.to(torch.uint8))  # Convertir de vuelta a uint8

    return interpolated_frames

def normalize_frames(frames, target_frame_count=15):
    """
    Ajusta la cantidad de frames a target_frame_count, ya sea interpolando o eliminando frames.
    """
    current_frame_count = len(frames)
    if current_frame_count < target_frame_count:
        return interpolate_frames(frames, target_frame_count)
    elif current_frame_count > target_frame_count:
        step = current_frame_count / target_frame_count
        indices = torch.arange(0, current_frame_count, step).long()[:target_frame_count]
        return [frames[i] for i in indices]
    else:
        return frames

def process_directory(word_directory, target_frame_count=15):
    """
    Procesa todas las muestras dentro de un directorio de palabra.
    """
    print(f"Procesando directorio: {word_directory}")
    for sample_name in os.listdir(word_directory):
        sample_directory = os.path.join(word_directory, sample_name)
        if os.path.isdir(sample_directory):
            print(f"Procesando muestra: {sample_name}")
            frames = read_frames_from_directory(sample_directory)
            if frames:
                normalized_frames = normalize_frames(frames, target_frame_count)
                clear_directory(sample_directory)
                save_normalized_frames(sample_directory, normalized_frames)

def save_normalized_frames(directory, frames):
    """
    Guarda los frames normalizados en el directorio especificado.
    """
    for i, frame in enumerate(frames, start=1):
        # Convertir los frames a uint8 (rango 0-255) de manera explícita
        frame_uint8 = frame.numpy().clip(0, 255).astype(np.uint8)  # Aseguramos que los valores estén en el rango correcto
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)  # Convertir de vuelta a BGR para OpenCV
        cv2.imwrite(os.path.join(directory, f'frame_{i:02}.jpg'), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])

def clear_directory(directory):
    """
    Elimina todos los archivos dentro de un directorio.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

if __name__ == "__main__":
    # OBTENER TODAS LAS PALABRAS A PROCESAR
    word_ids = [word for word in os.listdir(MUESTRAS_PATH) if os.path.isdir(os.path.join(MUESTRAS_PATH, word))]
    print(f"Palabras encontradas: {word_ids}")  # Imprimir palabras encontradas

    for word_id in word_ids:
        word_path = os.path.join(MUESTRAS_PATH, word_id)
        print(f'Normalizando frames para "{word_id}"...')
        process_directory(word_path, MODEL_FRAMES)
