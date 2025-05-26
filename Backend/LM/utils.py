import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from constants import KEYPOINTS_PATH

class SignLanguageDataset(Dataset):
    """Dataset para datos de lenguaje de señas"""
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def prepare_data():
    """
    Carga y prepara los datos de entrenamiento.
    
    Returns:
        sequences: Array de secuencias de keypoints
        labels: Array de etiquetas numéricas
        label_encoder: Codificador de etiquetas
    """
    sequences = []
    labels = []
    
    # Verificar si la carpeta de keypoints existe
    print(f"Buscando datos en: {KEYPOINTS_PATH}")
    if not os.path.exists(KEYPOINTS_PATH):
        print(f"Error: La carpeta de keypoints {KEYPOINTS_PATH} no existe.")
        return np.array([]), np.array([]), LabelEncoder()
    
    # Listar contenido de la carpeta para depuración
    print(f"Contenido de la carpeta {KEYPOINTS_PATH}:")
    for item in os.listdir(KEYPOINTS_PATH):
        print(f"  - {item}")
    
    # Recorrer carpetas de palabras dentro de la carpeta keypoints
    for word_folder in os.listdir(KEYPOINTS_PATH):
        word_path = os.path.join(KEYPOINTS_PATH, word_folder)
        
        # Verificar si es una carpeta
        if not os.path.isdir(word_path):
            continue
        
        print(f"Procesando carpeta: {word_folder}")
        
        # Buscar archivos H5 en la carpeta
        h5_files = [f for f in os.listdir(word_path) if f.endswith('.h5')]
        print(f"  Encontrados {len(h5_files)} archivos H5")
        
        # Procesar cada archivo H5
        for h5_file in h5_files:
            file_path = os.path.join(word_path, h5_file)
            
            try:
                with h5py.File(file_path, 'r') as f:
                    # Cargar datos del archivo H5
                    data = np.array(f['keypoints'])
                    print(f"Archivo cargado: {h5_file}, forma: {data.shape}")
                    
                    # Añadir secuencias y etiquetas
                    sequences.append(data)
                    labels.extend([word_folder] * len(data))
            except Exception as e:
                print(f"Error al cargar {file_path}: {str(e)}")
    
    # Convertir listas a arrays
    if sequences:
        # Concatenar todas las secuencias
        sequences = np.vstack([seq for seq in sequences])
        labels = np.array(labels)
        
        print(f"Total de secuencias cargadas: {len(sequences)}")
        print(f"Total de etiquetas procesadas: {len(labels)}")
        
        # Codificar etiquetas
        label_encoder = LabelEncoder()
        numeric_labels = label_encoder.fit_transform(labels)
        
        print(f"Clases encontradas: {label_encoder.classes_}")
        
        return sequences, numeric_labels, label_encoder
    else:
        return np.array([]), np.array([]), LabelEncoder()

def validate_model(model, val_loader, device):
    """
    Valida el modelo con el conjunto de validación.
    
    Args:
        model: Modelo PyTorch a validar
        val_loader: DataLoader con datos de validación
        device: Dispositivo donde ejecutar la validación
        
    Returns:
        Precisión de validación (%)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)  # Agregar dimensión de canal
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy
