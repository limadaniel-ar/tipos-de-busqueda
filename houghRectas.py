import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_lineas(ruta_imagen: str, 
                   umbral_bordes: tuple = (50, 150), 
                   umbral_lineas: int = 100,
                   filtro_orientacion: str = None):
    """
    Detecta líneas en una imagen usando Transformada de Hough con optimizaciones
    
    Args:
        ruta_imagen: Ruta de la imagen
        umbral_bordes: Umbrales para detección de bordes (bajo, alto)
        umbral_lineas: Umbral para detección de líneas
        filtro_orientacion: None, 'horizontal', 'vertical' o 'ambas'
    """
    # Cargar imagen con manejo seguro de rutas en Windows
    imagen = cv2.imread(ruta_imagen.replace('\\', '/'))
    
    if imagen is None:
        raise FileNotFoundError(f"Error: No se pudo cargar {ruta_imagen}")
    
    # 1. Preprocesamiento mejorado
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris_desenfoque = cv2.GaussianBlur(gris, (5, 5), 0)  # Reducción de ruido
    
    # 2. Detección de bordes optimizada
    bordes = cv2.Canny(gris_desenfoque, *umbral_bordes, apertureSize=3)
    
    # 3. Detección de líneas con HoughLinesP (probabilístico)
    lineas = cv2.HoughLinesP(
        bordes, 
        rho=1, 
        theta=np.pi/180, 
        threshold=umbral_lineas, 
        minLineLength=50,  
        maxLineGap=10     
    )
    
    # 4. Filtrado por orientación (opcional)
    lineas_filtradas = []
    if lineas is not None:
        if filtro_orientacion:
            for linea in lineas:
                x1, y1, x2, y2 = linea[0]
                dx, dy = x2 - x1, y2 - y1
                angulo = np.degrees(np.arctan2(dy, dx)) % 180
                
                if filtro_orientacion == 'horizontal' and (angulo < 15 or angulo > 165):
                    lineas_filtradas.append(linea)
                elif filtro_orientacion == 'vertical' and (75 < angulo < 105):
                    lineas_filtradas.append(linea)
                elif filtro_orientacion == 'ambas' and ((angulo < 15 or angulo > 165) or (75 < angulo < 105)):
                    lineas_filtradas.append(linea)
        else:
            lineas_filtradas = lineas
    
    # 5. Dibujar resultados
    imagen_lineas = imagen.copy()
    if lineas_filtradas:
        for linea in lineas_filtradas:
            x1, y1, x2, y2 = linea[0]
            cv2.line(imagen_lineas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return imagen, imagen_lineas, bordes

# Ejecución principal ##########################################
try:
    # Parámetros configurables
    RUTA_IMAGEN = r'C:\hop.png'  # Usar raw-string para Windows
    UMBRAL_BORDES = (150, 150)     # Ajustar según necesidad
    UMBRAL_LINEAS = 10           # Ajustar sensibilidad
    FILTRO = 'ambas'              # None, 'horizontal', 'vertical', 'ambas'
    
    # Procesamiento
    original, resultado, bordes = detectar_lineas(
        ruta_imagen=RUTA_IMAGEN,
        umbral_bordes=UMBRAL_BORDES,
        umbral_lineas=UMBRAL_LINEAS,
        filtro_orientacion=FILTRO
    )
    
    # Visualización profesional
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Imagen original
    ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original')
    ax[0].axis('off')
    
    # Bordes detectados
    ax[1].imshow(bordes, cmap='gray')
    ax[1].set_title(f'Bordes (Canny: {UMBRAL_BORDES})')
    ax[1].axis('off')
    
    # Resultado final
    ax[2].imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
    ax[2].set_title(f'Líneas Detectadas ({FILTRO.capitalize() if FILTRO else "Todas"})')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error: {str(e)}")
