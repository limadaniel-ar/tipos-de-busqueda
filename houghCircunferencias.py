import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_circunferencias(ruta_imagen: str, 
                              dp: float = 1.1, 
                              distancia_minima: int = 30, 
                              umbral_bordes: tuple = (150, 150), 
                              umbral_deteccion: int = 61, 
                              radio_minimo: int = 15, 
                              radio_maximo: int = 60):
    """
    Detecta circunferencias en una imagen usando la Transformada de Hough.
    
    Args:
        ruta_imagen: Ruta de la imagen a procesar.
        dp: Inverso de la resolución de la acumulación.
        distancia_minima: Distancia mínima entre el centro de las circunferencias detectadas.
        umbral_bordes: Tupla de umbrales para el detector de bordes de Canny.
        umbral_deteccion: Umbral para la detección de circunferencias.
        radio_minimo: Radio mínimo de las circunferencias a detectar.
        radio_maximo: Radio máximo de las circunferencias a detectar.
    """
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen.replace('\\', '/'))
    if imagen is None:
        raise FileNotFoundError(f"Error: No se pudo cargar la imagen desde {ruta_imagen}")
    
    # Procesamiento en un solo paso
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen_gris = cv2.medianBlur(imagen_gris, 5)
    
    # Detección de bordes
    bordes = cv2.Canny(imagen_gris, *umbral_bordes)
    
    # Detección de circunferencias
    circunferencias = cv2.HoughCircles(
        image=bordes,
        method=cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=distancia_minima,
        param1=umbral_bordes[1],
        param2=umbral_deteccion,
        minRadius=radio_minimo,
        maxRadius=radio_maximo
    )
    
    # Visualización integrada
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    
    # Imagen original
    ax[0].imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original')
    ax[0].axis('off')
    
    # Imagen con bordes
    ax[1].imshow(bordes, cmap='gray')
    ax[1].set_title(f'Bordes (Canny: {umbral_bordes})')
    ax[1].axis('off')
    
    # Imagen con detecciones
    resultado = imagen.copy()
    if circunferencias is not None:
        circunferencias = np.uint16(np.around(circunferencias))
        for (x, y, r) in circunferencias[0]:
            cv2.circle(resultado, (x, y), r, (0, 255, 0), 2)  # Circunferencia verde
            cv2.circle(resultado, (x, y), 2, (0, 0, 255), 3)  # Centro rojo
    
    ax[2].imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Circunferencias Detectadas')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Uso
try:
    detectar_circunferencias(r'C:\hop.png')
except Exception as e:
    print(str(e))
