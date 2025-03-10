import os
import numpy as np
import matplotlib.pyplot as plt
from constants import MODEL_FOLDER_PATH

def plot_training_history():
    """Visualiza el historial de entrenamiento"""
    # Cargar historial de entrenamiento
    history_path = os.path.join(MODEL_FOLDER_PATH, 'training_history.npy')
    
    try:
        history = np.load(history_path, allow_pickle=True).item()
        print("Historial de entrenamiento cargado correctamente")
    except Exception as e:
        print(f"Error al cargar el historial: {str(e)}")
        return
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Graficar pérdida
    ax1.plot(history['train_loss'])
    ax1.set_title('Pérdida durante el entrenamiento')
    ax1.set_ylabel('Pérdida')
    ax1.set_xlabel('Época')
    ax1.grid(True)
    
    # Graficar precisión
    ax2.plot(history['train_acc'], label='Entrenamiento')
    ax2.plot(history['val_acc'], label='Validación')
    ax2.set_title('Precisión durante el entrenamiento')
    ax2.set_ylabel('Precisión (%)')
    ax2.set_xlabel('Época')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Guardar figura
    output_path = os.path.join(MODEL_FOLDER_PATH, 'training_history.png')
    plt.savefig(output_path)
    print(f"Gráfico guardado en {output_path}")
    
    # Mostrar figura
    plt.show()

def main():
    # Verificar si existe el historial
    history_path = os.path.join(MODEL_FOLDER_PATH, 'training_history.npy')
    if not os.path.exists(history_path):
        print(f"Error: No se encontró el archivo de historial en {history_path}")
        return
    
    # Visualizar historial
    plot_training_history()

if __name__ == "__main__":
    main()
