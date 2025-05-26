import os
import torch
import h5py
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, get_keypoints
from constants import ROOT_PATH, MUESTRAS_PATH, KEYPOINTS_PATH

def create_keypoints(word_id, words_path, fixed_length=3):
    """
    ### CREAR KEYPOINTS PARA UNA PALABRA
    Recorre la carpeta de frames de la palabra y guarda sus keypoints en un archivo `.h5` dentro de la subcarpeta de la palabra.
    
    - `word_id`: Nombre de la palabra.
    - `words_path`: Ruta base donde están las muestras de la palabra.
    - `fixed_length`: Longitud fija para las secuencias de keypoints.
    """
    keypoints_data = []
    frames_path = os.path.join(words_path, word_id)
    
    # Crear subcarpeta para la palabra en `KEYPOINTS_PATH`
    word_folder_path = os.path.join(KEYPOINTS_PATH, word_id)
    create_folder(word_folder_path)
    
    # Inicializar Holistic
    with Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        print(f'🔹 Creando keypoints de "{word_id}"...')
        if not os.path.exists(frames_path):
            print(f"❌ Directorio no encontrado: {frames_path}")
            return
        
        sample_list = os.listdir(frames_path)
        sample_count = len(sample_list)
        
        for n_sample, sample_name in enumerate(sample_list, start=1):
            sample_path = os.path.join(frames_path, sample_name)
            
            # Extraer keypoints como tensor de PyTorch
            try:
                keypoints_sequence = get_keypoints(holistic, sample_path)
                
                # Verificar si no se han extraído keypoints válidos
                if keypoints_sequence is None or len(keypoints_sequence) == 0:
                    print(f"❌ No se encontraron keypoints en {sample_name}")
                    continue
                
                # Convertir a tensor y verificar que la longitud es consistente
                keypoints_tensor = torch.tensor(keypoints_sequence, dtype=torch.float32)
                
                if keypoints_tensor.numel() == 0:
                    print(f"❌ Keypoints vacíos en {sample_name}")
                    continue
                
                # Asegurar que todas las secuencias tengan la misma longitud
                if keypoints_tensor.size(0) > fixed_length:
                    keypoints_tensor = keypoints_tensor[:fixed_length, :]  # Recortar a la longitud deseada
                elif keypoints_tensor.size(0) < fixed_length:
                    padding = torch.zeros((fixed_length - keypoints_tensor.size(0), keypoints_tensor.size(1)))
                    keypoints_tensor = torch.cat([keypoints_tensor, padding], dim=0)
                
                keypoints_data.append(keypoints_tensor.flatten().numpy())
                print(f"✅ {n_sample}/{sample_count} procesados", end="\r")
            except Exception as e:
                print(f"Error procesando {sample_name}: {e}")
                continue
        
        # Si se procesaron keypoints, se guarda el archivo .h5 en la subcarpeta
        if keypoints_data:
            keypoints_data = torch.tensor(keypoints_data, dtype=torch.float32)
            hdf_path = os.path.join(word_folder_path, f"{word_id}.h5")
            
            # Guardar los keypoints en un archivo HDF5
            with h5py.File(hdf_path, 'w') as f:
                f.create_dataset('keypoints', data=keypoints_data.numpy())  # Guardar como dataset

            print(f"✔️ Keypoints guardados en {hdf_path} ({len(keypoints_data)} muestras procesadas)\n")
        else:
            print(f"❌ No se procesaron keypoints para la palabra: {word_id}\n")

if __name__ == "__main__":
    # Crear la carpeta `keypoints` si no existe
    create_folder(KEYPOINTS_PATH)
    
    # Lista de palabras a procesar
    if not os.path.exists(MUESTRAS_PATH):
        print(f"❌ No se encontró la carpeta: {MUESTRAS_PATH}")
    else:
        word_ids = [word for word in os.listdir(MUESTRAS_PATH) if os.path.isdir(os.path.join(MUESTRAS_PATH, word))]
        
        for word_id in word_ids:
            create_keypoints(word_id, MUESTRAS_PATH, fixed_length=3)  # Fijamos la longitud de 3 frames
