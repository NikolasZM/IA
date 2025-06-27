import numpy as np

x = np.fromfile("X_test.bin", dtype=np.float32)
y = np.fromfile("y_test.bin", dtype=np.float32)  

print("Tamaño X_train.bin (total floats):", x.shape[0])
print("Tamaño Y_train.bin (total floats):", y.shape[0])

n_images = x.shape[0] // (48 * 48)
print("Cantidad de imágenes:", n_images)
print("Cantidad de etiquetas:", y.shape[0])

if n_images == y.shape[0]:
    print("Coinciden.")
else:
    print("No coinciden.")
