import numpy as np
import matplotlib.pyplot as plt

class RedHopfield:
    def __init__(self, tamaño):
        self.tamaño = tamaño
        self.pesos = np.zeros((tamaño, tamaño))

    def entrenar(self, patrones):
        for p in patrones:
            self.pesos += np.outer(p, p)
        np.fill_diagonal(self.pesos, 0)

    def predecir(self, patron_entrada, pasos=5):
        patron = patron_entrada.copy()
        historial_patrones = [patron.copy()]  # Guardar el estado inicial
        for _ in range(pasos):
            for i in range(self.tamaño):
                s = np.dot(self.pesos[i], patron)
                patron[i] = 1 if s > 0 else 0
            historial_patrones.append(patron.copy())  # Guardar cada paso
        return historial_patrones

# Crear una imagen de ejemplo (10x10)
original = np.array([
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Función para agregar ruido aleatorio
def agregar_ruido_aleatorio(imagen, nivel_ruido=0.1):
    imagen_con_ruido = imagen.copy()
    num_pixeles_ruido = int(nivel_ruido * imagen.size)
    for _ in range(num_pixeles_ruido):
        x = np.random.randint(0, imagen.shape[0])
        y = np.random.randint(0, imagen.shape[1])
        imagen_con_ruido[x, y] = 1 if imagen_con_ruido[x, y] == 0 else 0  # Cambiar de 0 a 1 o de 1 a 0
    return imagen_con_ruido

# Agregar ruido a la imagen original (10% de píxeles con ruido)
imagen_con_ruido = agregar_ruido_aleatorio(original, nivel_ruido=0.1)

# Entrenar la red con el patrón original
red_hopfield = RedHopfield(tamaño=100)
red_hopfield.entrenar([original.flatten()])

# Predecir el patrón limpio y guardar los pasos
historial_patrones = red_hopfield.predecir(imagen_con_ruido.flatten(), pasos=10)

# Mostrar resultados en una animación
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Imagen Original')
plt.imshow(original, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Imagen con Ruido')
img_plot = plt.imshow(imagen_con_ruido, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.ion()  # Activar modo interactivo

# Animación de limpieza del patrón
for paso in historial_patrones:
    # Realizar múltiples actualizaciones en cada iteración
    for _ in range(5):  # Ajustar el número de actualizaciones por paso
        img_plot.set_array(paso.reshape(10, 10))  # Actualizar la imagen
        plt.pause(0.2)  # Pausa corta para cada actualización
    plt.pause(1)  # Pausa de 1 segundo después de cada paso completo

plt.ioff()  # Desactivar modo interactivo
plt.show()
