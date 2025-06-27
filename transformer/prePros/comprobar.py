import numpy as np

# Parámetros según tus datos
num_samples = 28709       # Cambia esto al número de muestras reales
image_size = 48 * 48      # 2304

# Verifica imágenes
X_check = np.fromfile("prePros/X_train.bin", dtype=np.float32)
X_check = X_check.reshape((num_samples, image_size))

print("Shape de X_train:", X_check.shape)
print("Primer vector de pixeles normalizado (X[0]):")
print(X_check[0])

# Verifica etiquetas
y_check = np.fromfile("prePros/y_train.bin", dtype=np.uint8)
print("\nShape de y_train:", y_check.shape)
print("Primeras 10 etiquetas:", y_check[:10])
