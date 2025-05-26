import torch
import torch.nn as nn
import os
from constants import LENGTH_KEYPOINTS, MODEL_FRAMES

class SignLanguageCNN(nn.Module):
    """
    Modelo CNN para clasificación de lenguaje de señas.
    
    Procesa secuencias de keypoints y predice la palabra/signo correspondiente.
    """
    def __init__(self, input_size=None, num_classes=20):
        super(SignLanguageCNN, self).__init__()
        
        # Si no se proporciona un tamaño de entrada, usar el de las constantes
        if input_size is None:
            input_size = LENGTH_KEYPOINTS * MODEL_FRAMES
        
        # Características de entrada
        self.input_size = input_size
        
        # Capas convolucionales
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        
        # Capas de pooling y activación
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Calculamos el tamaño de la salida después de las convoluciones y pooling
        # Después de 3 capas de pooling, el tamaño se reduce a 1/8
        self.flattened_size = 128 * (input_size // 8)
        
        # Capas fully connected con tamaño dinámico
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        print(f"Modelo inicializado con tamaño de entrada: {input_size}")
        print(f"Tamaño después de convoluciones y pooling: {self.flattened_size}")

    def forward(self, x):
        # x tiene forma (batch_size, 1, input_size)
        batch_size = x.size(0)
        
        # Aplicar convoluciones
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        # Aplanar para las capas fully connected
        x = x.view(batch_size, -1)
        
        # Aplicar capas fully connected
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def save(self, path):
        """Guarda el modelo en la ruta especificada"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Modelo guardado en: {path}")
        
    def load(self, path):
        """Carga los pesos del modelo desde la ruta especificada"""
        self.load_state_dict(torch.load(path))
        self.eval()
        print(f"Modelo cargado desde: {path}")
