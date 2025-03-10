import os
import cv2

# SETTINGS
MIN_LENGTH_FRAMES = 3
LENGTH_KEYPOINTS = 54
MODEL_FRAMES = 18

# PATHS
ROOT_PATH = os.getcwd()
BASE_PATH = os.path.join(ROOT_PATH, "backend")
LM_PATH = os.path.join(BASE_PATH, "LM")
MUESTRAS_PATH = os.path.join(LM_PATH, "muestras")  # Corregido para que coincida con la carpeta "muestras"
DATA_PATH = os.path.join(LM_PATH, "data")
KEYPOINTS_PATH = os.path.join(DATA_PATH, "keypoints")  # Corregido para que apunte correctamente a keypoints
MODEL_FOLDER_PATH = os.path.join(LM_PATH, "models")

# Archivos específicos
DATA_JSON_PATH = os.path.join(DATA_PATH, "data.json")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}.keras")
WORDS_JSON_PATH = os.path.join(MODEL_FOLDER_PATH, "words.json")

# SHOW IMAGE PARAMETERS
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

# WORDS TEXT MAPPING
words_text = {
    "adios": "ADIOS",
    "bien": "BIEN",
    "buenas_noches": "BUENAS NOCHES",
    "buenas_tardes": "BUENAS TARDES",
    "buenos_dias": "BUENOS DÍAS",
    "como_estas": "¿CÓMO ESTÁS?",
    "disculpa": "DISCULPA",
    "gracias": "GRACIAS",
    "hola": "HOLA",
    "mal": "MAL",
    "mas_o_menos": "MÁS O MENOS",
    "me_ayudas": "¿ME AYUDAS?",
    "por_favor": "POR FAVOR",
    "lo_siento": "LO SIENTO",
    "te_quiero": "TE QUIERO",
    "ayuda": "AYUDA",
    "permiso": "PERMISO",
    "felicidades": "FELICIDADES",
    "perdon": "PERDÓN",
    "amor": "AMOR",
}
