def buscar_punto_a(punto_inicial, punto_a, rango_busqueda):
    """
    Realiza una búsqueda exhaustiva alternando entre izquierda y derecha para encontrar el punto A.
    
    :param punto_inicial: La posición inicial del autómata (B).
    :param punto_a: La posición del punto que se busca (A).
    :param rango_busqueda: El rango máximo de búsqueda en ambas direcciones.
    :return: La posición del punto A si se encuentra, de lo contrario, None.
    """
    contador = 1
    for i in range(1, rango_busqueda + 1):
        # Explorar hacia la izquierda
        posicion_izquierda = punto_inicial - i
        print(f"Explorando hacia la izquierda: B{contador}= {posicion_izquierda}")
        contador = contador + 1
        if posicion_izquierda == punto_a:
            return posicion_izquierda
        
        # Explorar hacia la derecha
        posicion_derecha = punto_inicial + i
        print(f"Explorando hacia la derecha: B{contador}= {posicion_derecha}")
        contador = contador + 1
        if posicion_derecha == punto_a:
            return posicion_derecha
    
    return None

# Ejemplo de uso
punto_inicial = 0  # Posición B
punto_a = 3        # Posición A que se busca
rango_busqueda = 5 # Rango máximo de búsqueda

resultado = buscar_punto_a(punto_inicial, punto_a, rango_busqueda)

if resultado is not None:
    print(f"El punto A se encuentra en: {resultado}")
else:
    print("El punto A no se encontró en el rango especificado.")