import os
import cv2
import torch
import numpy as np
from datetime import datetime
import mediapipe as mp
from helpers import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, MUESTRAS_PATH, ROOT_PATH  # Se corrigió FRAME_ACTIONS_PATH por MUESTRAS_PATH


mp_holistic = mp.solutions.holistic

def capture_samples(path, margin_frame=1, min_cant_frames=5, delay_frames=3):
   
    create_folder(path)

    count_frame = 0
    frames = []
    fix_frames = 0
    recording = False

    # Cargar modelo Holistic de MediaPipe
    with mp_holistic.Holistic() as holistic_model:
        video = cv2.VideoCapture(0)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            image = frame.copy()
            results = mediapipe_detection(frame, holistic_model)

            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    cv2.putText(image, 'Capturando...', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                    frames.append(torch.tensor(frame, dtype=torch.uint8))  # Convertir frame a tensor
            else:
                if len(frames) >= min_cant_frames + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    frames = frames[: -(margin_frame + delay_frames)]
                    today = datetime.now().strftime('%y%m%d%H%M%S%f')
                    output_folder = os.path.join(path, f"sample_{today}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)

                recording, fix_frames = False, 0
                frames, count_frame = [], 0
                cv2.putText(image, 'Listo para capturar...', FONT_POS, FONT, FONT_SIZE, (0, 220, 100))

            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    word_name = input("Ingrese el nombre de la palabra a capturar: ")
    word_path = os.path.join(ROOT_PATH, MUESTRAS_PATH, word_name)  # Se corrigió FRAME_ACTIONS_PATH por MUESTRAS_PATH
    capture_samples(word_path)
