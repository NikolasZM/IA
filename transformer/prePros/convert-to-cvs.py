import os
import cv2
import csv
from tqdm import tqdm

emotion_labels = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

def contar_imagenes(root_dir):
    total = 0
    for emocion in emotion_labels:
        carpeta = os.path.join(root_dir, emocion)
        if os.path.isdir(carpeta):
            total += len(os.listdir(carpeta))
    return total

def procesar_directorio_y_guardar(root_dir, nombre_csv):
    total_imgs = contar_imagenes(root_dir)
    print(f"\nProcesando {total_imgs} imágenes desde: {root_dir}")

    with open(nombre_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        for emocion, label in emotion_labels.items():
            carpeta = os.path.join(root_dir, emocion)
            if not os.path.isdir(carpeta):
                continue

            archivos = os.listdir(carpeta)
            for archivo in tqdm(archivos, desc=f"{nombre_csv} - {emocion}", leave=False):
                path = os.path.join(carpeta, archivo)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None or img.shape != (48, 48):
                    continue

                fila = [label] + img.flatten().tolist()
                writer.writerow(fila)

    print(f"✅ Guardado en {nombre_csv}")

train_dir = "train"
test_dir = "test"

procesar_directorio_y_guardar(train_dir, "fer2013_train.csv")
procesar_directorio_y_guardar(test_dir, "fer2013_test.csv")