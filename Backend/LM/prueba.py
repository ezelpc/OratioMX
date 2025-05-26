import os
import shutil
from constants import DATA_PATH, KEYPOINTS_PATH, MODEL_FOLDER_PATH, words_text

def setup_folders():
    """Configura la estructura de carpetas necesaria para el proyecto"""
    print("=== Configurando estructura de carpetas ===")
    
    # Verificar y crear carpeta de datos principal
    if not os.path.exists(DATA_PATH):
        print(f"Creando carpeta de datos: {DATA_PATH}")
        os.makedirs(DATA_PATH, exist_ok=True)
    else:
        print(f"La carpeta de datos ya existe: {DATA_PATH}")
    
    # Verificar y crear carpeta de keypoints
    if not os.path.exists(KEYPOINTS_PATH):
        print(f"Creando carpeta de keypoints: {KEYPOINTS_PATH}")
        os.makedirs(KEYPOINTS_PATH, exist_ok=True)
    else:
        print(f"La carpeta de keypoints ya existe: {KEYPOINTS_PATH}")
    
    # Crear subcarpetas para cada palabra dentro de keypoints
    for word in words_text.keys():
        word_folder = os.path.join(KEYPOINTS_PATH, word)
        if not os.path.exists(word_folder):
            print(f"Creando carpeta para '{word}': {word_folder}")
            os.makedirs(word_folder, exist_ok=True)
    
    # Verificar y crear carpeta de modelos
    if not os.path.exists(MODEL_FOLDER_PATH):
        print(f"Creando carpeta de modelos: {MODEL_FOLDER_PATH}")
        os.makedirs(MODEL_FOLDER_PATH, exist_ok=True)
    else:
        print(f"La carpeta de modelos ya existe: {MODEL_FOLDER_PATH}")
    
    # Verificar si hay una estructura incorrecta y corregirla
    incorrect_path = os.path.join(KEYPOINTS_PATH, "data")
    if os.path.exists(incorrect_path) and os.path.isdir(incorrect_path):
        print(f"\n¡ATENCIÓN! Se encontró una estructura incorrecta: {incorrect_path}")
        
        # Verificar si hay una carpeta "keypoints" dentro de la estructura incorrecta
        nested_keypoints = os.path.join(incorrect_path, "keypoints")
        if os.path.exists(nested_keypoints) and os.path.isdir(nested_keypoints):
            print(f"Se encontró una carpeta keypoints anidada: {nested_keypoints}")
            
            # Mover contenido a la ubicación correcta
            for item in os.listdir(nested_keypoints):
                src_path = os.path.join(nested_keypoints, item)
                dst_path = os.path.join(KEYPOINTS_PATH, item)
                
                if not os.path.exists(dst_path):
                    print(f"Moviendo {item} a la ubicación correcta")
                    try:
                        shutil.move(src_path, dst_path)
                    except Exception as e:
                        print(f"Error al mover {item}: {str(e)}")
            
            # Eliminar estructura incorrecta
            try:
                shutil.rmtree(incorrect_path)
                print(f"Estructura incorrecta eliminada: {incorrect_path}")
            except Exception as e:
                print(f"Error al eliminar estructura incorrecta: {str(e)}")
    
    print("\n=== Estructura de carpetas configurada ===")
    print(f"Carpeta de datos: {os.path.abspath(DATA_PATH)}")
    print(f"Carpeta de keypoints: {os.path.abspath(KEYPOINTS_PATH)}")
    print(f"Carpeta de modelos: {os.path.abspath(MODEL_FOLDER_PATH)}")
    
    # Verificar si hay archivos H5 en las carpetas
    total_h5_files = 0
    for word in words_text.keys():
        word_folder = os.path.join(KEYPOINTS_PATH, word)
        if os.path.exists(word_folder):
            h5_files = [f for f in os.listdir(word_folder) if f.endswith('.h5')]
            if h5_files:
                print(f"  '{word}': {len(h5_files)} archivos H5")
                total_h5_files += len(h5_files)
    
    if total_h5_files == 0:
        print("\n¡ATENCIÓN! No se encontraron archivos H5 en las carpetas.")
        print("Debes agregar archivos H5 con datos de keypoints antes de entrenar el modelo.")
    else:
        print(f"\nTotal de archivos H5 encontrados: {total_h5_files}")

if __name__ == "__main__":
    setup_folders()
