import os
import json
import torch
import numpy as np
import h5py
from model import SignLanguageCNN
from constants import MODEL_FOLDER_PATH, WORDS_JSON_PATH, DATA_PATH, MODEL_PATH

def load_model_and_classes():
    """Carga el modelo de lenguaje de señas y el mapeo de clases"""
    # Verificar existencia del archivo JSON
    if not os.path.exists(WORDS_JSON_PATH):
        print(f"Error: No se encontró {WORDS_JSON_PATH}")
        return None, None

    # Cargar mapeo de clases
    try:
        with open(WORDS_JSON_PATH, 'r') as f:
            class_mapping = json.load(f)
        num_classes = len(class_mapping)
        print(f"Mapeo de clases cargado con {num_classes} clases.")
    except Exception as e:
        print(f"Error al cargar el mapeo de clases: {str(e)}")
        return None, None

    # Verificar existencia del modelo
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        return None, None

    try:
        # Determinar input_size a partir de los datos disponibles
        input_size = 4986  # Valor por defecto
        for word_folder in os.listdir(DATA_PATH):
            word_path = os.path.join(DATA_PATH, word_folder)
            if os.path.isdir(word_path):
                h5_files = [f for f in os.listdir(word_path) if f.endswith('.h5')]
                if h5_files:
                    file_path = os.path.join(word_path, h5_files[0])
                    with h5py.File(file_path, 'r') as f:
                        data = np.array(f['keypoints'])
                        input_size = data.shape[1]
                        break

        # Cargar el modelo
        model = SignLanguageCNN(input_size=input_size, num_classes=num_classes)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print(f"Modelo cargado desde {MODEL_PATH}")

        return model, class_mapping
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        return None, None

def load_test_sequence(file_path):
    """Carga una secuencia de prueba desde un archivo H5"""
    if not os.path.exists(file_path):
        print(f"Error: No se encontró {file_path}")
        return None

    try:
        with h5py.File(file_path, 'r') as f:
            data = np.array(f['keypoints'])
            return data
    except Exception as e:
        print(f"Error al cargar la secuencia de prueba: {str(e)}")
        return None

def predict(model, sequence, class_mapping):
    """Realiza una predicción con el modelo entrenado"""
    if len(sequence.shape) == 1:
        sequence = sequence.reshape(1, -1)

    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        outputs = model(sequence_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    predicted_label = class_mapping[str(predicted_class)]

    return predicted_label, probabilities.numpy()[0]

def main():
    model, class_mapping = load_model_and_classes()
    
    if model is None or class_mapping is None:
        print("No se pudo cargar el modelo o el mapeo de clases.")
        return
    
    print(f"Modelo cargado con {len(class_mapping)} clases")

    # Mostrar archivos disponibles
    available_files = []
    for word_folder in os.listdir(DATA_PATH):
        word_path = os.path.join(DATA_PATH, word_folder)
        if os.path.isdir(word_path):
            for h5_file in [f for f in os.listdir(word_path) if f.endswith('.h5')]:
                file_path = os.path.join(word_path, h5_file)
                available_files.append((file_path, f"{word_folder}/{h5_file}"))

    if not available_files:
        print("No hay archivos disponibles para predicción.")
        return

    for i, (_, file_name) in enumerate(available_files):
        print(f"{i+1}. {file_name}")

    try:
        choice = int(input("\nSelecciona un archivo (número): ")) - 1
        if 0 <= choice < len(available_files):
            file_path, file_name = available_files[choice]
            print(f"Archivo seleccionado: {file_name}")

            sequence = load_test_sequence(file_path)

            if sequence is None:
                print("No se pudo cargar la secuencia.")
                return

            print(f"Secuencia cargada con forma: {sequence.shape}")

            for i, frame in enumerate(sequence):
                predicted_label, probabilities = predict(model, frame, class_mapping)

                print(f"\nFrame {i+1}:")
                print(f"Predicción: {predicted_label}")
                for class_idx, prob in enumerate(probabilities):
                    label = class_mapping[str(class_idx)]
                    print(f"  {label}: {prob:.4f}")
        else:
            print("Selección inválida.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
