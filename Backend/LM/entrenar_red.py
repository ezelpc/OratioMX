import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

# Importar módulos propios
from model import SignLanguageCNN
from utils import SignLanguageDataset, prepare_data, validate_model
from constants import MODEL_FOLDER_PATH, WORDS_JSON_PATH, words_text

def train_model(model, train_loader, val_loader, device, epochs=50, learning_rate=0.001):
    """
    Entrena el modelo con los datos proporcionados.
    
    Args:
        model: Modelo PyTorch a entrenar
        train_loader: DataLoader con datos de entrenamiento
        val_loader: DataLoader con datos de validación
        device: Dispositivo donde ejecutar el entrenamiento (CPU/GPU)
        epochs: Número de épocas de entrenamiento
        learning_rate: Tasa de aprendizaje
        
    Returns:
        Historial de entrenamiento (pérdida y precisión)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Barra de progreso para el entrenamiento
        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)  # Agregar dimensión de canal
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass y optimización
            loss.backward()
            optimizer.step()
            
            # Estadísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            
            # Actualizar barra de progreso
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calcular métricas de entrenamiento
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        
        # Validación
        val_accuracy = validate_model(model, val_loader, device)
        
        # Actualizar scheduler
        scheduler.step(train_loss)
        
        # Guardar métricas en el historial
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        
        print(f"Época {epoch+1}/{epochs}, Pérdida: {train_loss:.4f}, "
              f"Precisión entrenamiento: {train_accuracy:.2f}%, Precisión validación: {val_accuracy:.2f}%")
        
        # Guardar el mejor modelo
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_path = os.path.join(MODEL_FOLDER_PATH, 'best_model.pth')
            try:
                # Asegurarse de que la carpeta existe
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f"Modelo guardado en {model_path} con precisión de validación: {val_accuracy:.2f}%")
            except Exception as e:
                print(f"Error al guardar el modelo: {str(e)}")
    
    return history

def main():
    # Crear carpeta para modelos si no existe
    os.makedirs(MODEL_FOLDER_PATH, exist_ok=True)
    print(f"Carpeta de modelos: {os.path.abspath(MODEL_FOLDER_PATH)}")
    
    # Configurar dispositivo (GPU si está disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Preparar datos
    sequences, labels, label_encoder = prepare_data()
    
    if len(sequences) == 0:
        print("Error: No se cargaron datos. Verificar la carpeta de datos.")
        return
    
    # Mostrar información sobre los datos
    print(f"Forma de las secuencias: {sequences.shape}")
    print(f"Forma de las etiquetas: {labels.shape}")
    
    # Contar muestras por clase
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Distribución de clases:")
    for i, label in enumerate(unique_labels):
        class_name = label_encoder.inverse_transform([label])[0]
        print(f"  {class_name}: {counts[i]} muestras")
    
    # Verificar si hay suficientes muestras para estratificar
    min_samples_per_class = min(counts)
    
    # Dividir en conjuntos de entrenamiento y validación
    if min_samples_per_class < 2:
        print("ADVERTENCIA: Algunas clases tienen menos de 2 muestras. No se puede usar estratificación.")
        print("Usando división simple sin estratificación.")
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, labels, test_size=0.2, random_state=42, stratify=labels
        )
    
    print(f"Conjunto de entrenamiento: {X_train.shape}, Conjunto de validación: {X_val.shape}")
    
    # Verificar si hay clases sin muestras en el conjunto de validación
    train_classes = set(y_train)
    val_classes = set(y_val)
    missing_classes = train_classes - val_classes
    
    if missing_classes:
        print(f"ADVERTENCIA: Algunas clases no están presentes en el conjunto de validación: {missing_classes}")
        print("Usando todo el conjunto de datos para entrenamiento y validación.")
        
        # En este caso, usamos los mismos datos para entrenamiento y validación
        X_train, y_train = sequences, labels
        X_val, y_val = sequences, labels
    
    # Crear datasets y dataloaders
    train_dataset = SignLanguageDataset(X_train, y_train)
    val_dataset = SignLanguageDataset(X_val, y_val)
    
    # Usar un batch_size más pequeño para conjuntos de datos pequeños
    batch_size = min(8, len(train_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Crear modelo con el tamaño de entrada correcto
    input_size = sequences.shape[1]  # Obtener el tamaño real de los datos
    num_classes = len(label_encoder.classes_)
    model = SignLanguageCNN(input_size=input_size, num_classes=num_classes).to(device)
    
    print(f"Modelo creado con {num_classes} clases")
    
    # Entrenar modelo
    history = train_model(model, train_loader, val_loader, device, epochs=100)
    
    # Guardar mapeo de etiquetas
    class_mapping = {str(i): label for i, label in enumerate(label_encoder.classes_)}
    
    # Guardar mapeo de clases a palabras
    try:
        os.makedirs(os.path.dirname(WORDS_JSON_PATH), exist_ok=True)
        with open(WORDS_JSON_PATH, 'w') as f:
            json.dump(class_mapping, f, indent=4)
        print(f"Mapeo de clases guardado en {WORDS_JSON_PATH}")
    except Exception as e:
        print(f"Error al guardar el mapeo de clases: {str(e)}")
    
    # Guardar historial de entrenamiento
    try:
        history_path = os.path.join(MODEL_FOLDER_PATH, 'training_history.npy')
        np.save(history_path, history)
        print(f"Historial de entrenamiento guardado en {history_path}")
    except Exception as e:
        print(f"Error al guardar el historial: {str(e)}")
    
    # Mostrar resultados finales
    print("\nEntrenamiento completado!")
    print(f"Mejor precisión de validación: {max(history['val_acc']):.2f}%")
    
    # Verificar archivos generados
    print("\nArchivos generados:")
    try:
        for file in os.listdir(MODEL_FOLDER_PATH):
            print(f"- {file}")
    except Exception as e:
        print(f"Error al listar archivos: {str(e)}")

if __name__ == '__main__':
    main()
