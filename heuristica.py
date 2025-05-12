import heapq
import numpy as np
from typing import List, Tuple, Dict, Set

class Nodo:
    """Representa un nodo en el espacio de búsqueda."""
    
    def __init__(self, estado, padre=None, accion=None, costo_camino=0):
        self.estado = estado  # Estado (posición en la cara del motor)
        self.padre = padre  # Nodo padre
        self.accion = accion  # Acción que llevó a este nodo
        self.costo_camino = costo_camino  # Costo del camino hasta este nodo
        self.profundidad = 0  # Profundidad en el árbol de búsqueda
        if padre:
            self.profundidad = padre.profundidad + 1
    
    def __lt__(self, otro):
        return self.costo_camino < otro.costo_camino
    
    def __eq__(self, otro):
        return self.estado == otro.estado
    
    def __hash__(self):
        return hash(self.estado)

def distancia_manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Calcula la distancia Manhattan entre dos puntos."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def busqueda_a_estrella(matriz_motor: np.ndarray, inicio: Tuple[int, int], objetivo: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Implementa el algoritmo A* para encontrar un camino en la cara del motor.
    
    Args:
        matriz_motor: Matriz numpy donde 0 representa espacio libre y 1 un obstáculo
        inicio: Tupla (fila, columna) con la posición inicial
        objetivo: Tupla (fila, columna) con la posición objetivo
    
    Returns:
        Lista de tuplas con las posiciones del camino encontrado
    """
    # Acciones posibles: arriba, derecha, abajo, izquierda
    direcciones = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    nombres_direcciones = ["arriba", "derecha", "abajo", "izquierda"]
    
    # Inicialización
    nodo_inicio = Nodo(inicio)
    if inicio == objetivo:
        return [inicio]
    
    # Frontera (nodos por explorar) implementada como cola de prioridad
    frontera = []
    # f(n) = g(n) + h(n) donde g es el costo del camino y h es la heurística
    valor_f = distancia_manhattan(inicio, objetivo)
    heapq.heappush(frontera, (valor_f, nodo_inicio))
    
    # Diccionario para mapear estados a nodos
    hash_frontera = {inicio: nodo_inicio}
    
    # Conjunto de estados explorados
    explorados = set()
    
    # Contador de nodos explorados
    nodos_explorados = 0
    
    while frontera:
        # Extraer nodo con menor f(n)
        f_actual, nodo_actual = heapq.heappop(frontera)
        estado_actual = nodo_actual.estado
        
        # Eliminar del diccionario de la frontera
        if estado_actual in hash_frontera:
            del hash_frontera[estado_actual]
        
        nodos_explorados += 1
        
        # Verificar si hemos llegado al objetivo
        if estado_actual == objetivo:
            # Reconstruir camino
            camino = []
            while nodo_actual:
                camino.append(nodo_actual.estado)
                nodo_actual = nodo_actual.padre
            return camino[::-1]  # Invertir para tener el camino de inicio a fin
        
        # Marcar como explorado
        explorados.add(estado_actual)
        
        # Expandir nodo actual
        for i, (dy, dx) in enumerate(direcciones):
            # Calcular nuevo estado
            nueva_fila, nueva_columna = estado_actual[0] + dy, estado_actual[1] + dx
            nuevo_estado = (nueva_fila, nueva_columna)
            
            # Verificar si es válido: dentro de los límites y no es obstáculo
            if (0 <= nueva_fila < matriz_motor.shape[0] and 
                0 <= nueva_columna < matriz_motor.shape[1] and 
                matriz_motor[nueva_fila, nueva_columna] == 0 and
                nuevo_estado not in explorados):
                
                # Crear nuevo nodo
                nuevo_costo_camino = nodo_actual.costo_camino + 1  # Costo uniforme
                nuevo_nodo = Nodo(nuevo_estado, nodo_actual, nombres_direcciones[i], nuevo_costo_camino)
                
                # Calcular f(n) = g(n) + h(n)
                nuevo_f = nuevo_costo_camino + distancia_manhattan(nuevo_estado, objetivo)
                
                # Si el estado no está en la frontera o encontramos un camino mejor
                if (nuevo_estado not in hash_frontera or 
                    nuevo_costo_camino < hash_frontera[nuevo_estado].costo_camino):
                    
                    # Actualizar o agregar a la frontera
                    hash_frontera[nuevo_estado] = nuevo_nodo
                    heapq.heappush(frontera, (nuevo_f, nuevo_nodo))
    
    # Si no encontramos camino
    return None

def crear_cara_motor(tamano=10, probabilidad_obstaculo=0.3, semilla=42, inicio: Tuple[int, int] = (0, 0)):
    """
    Crea una cara de motor aleatoria.
    
    Args:
        tamano: Tamaño de la cara del motor (cuadrado)
        probabilidad_obstaculo: Probabilidad de cada celda de ser un obstáculo
        semilla: Semilla para reproducibilidad
        inicio: Tupla (fila, columna) con la posición inicial, debe ser un espacio libre
    
    Returns:
        Matriz numpy donde 0 representa espacio libre y 1 obstáculo
    """
    np.random.seed(semilla)
    cara_motor = np.random.random((tamano, tamano)) < probabilidad_obstaculo
    cara_motor = cara_motor.astype(np.int32)
    
    # Asegurarse que el inicio y el final estén libres
    cara_motor[inicio] = 0  # Asegurarse que el inicio es libre
    cara_motor[5, 5] = 0  # Final
    
    # Asegurarse que existe al menos un camino del inicio al final
    # (simplemente creando un corredor)
    for i in range(tamano):
        cara_motor[i, 0] = 0
    for j in range(tamano):
        cara_motor[tamano-1, j] = 0
    
    return cara_motor

def imprimir_cara_motor(cara_motor: np.ndarray, camino: List[Tuple[int, int]] = None):
    """
    Imprime la cara del motor con el camino si se proporciona.
    
    Args:
        cara_motor: Matriz numpy donde 0 representa espacio libre y 1 obstáculo
        camino: Lista de tuplas con las posiciones del camino
    """
    simbolos = {
        0: ' ',  # Espacio libre
        1: '█',  # Obstáculo
    }
    
    conjunto_camino = set()
    if camino:
        conjunto_camino = set(camino)
    
    # Imprimir encabezado de columnas
    print('  ', end='')
    for j in range(cara_motor.shape[1]):
        print(f'{j%10}', end='')
    print()
    
    # Imprimir cara del motor
    for i in range(cara_motor.shape[0]):
        print(f'{i%10} ', end='')  # Número de fila
        for j in range(cara_motor.shape[1]):
            pos = (i, j)
            if pos in conjunto_camino:
                if pos == camino[0]:  # Inicio
                    print('B', end='')
                elif pos == camino[-1]:  # Fin
                    print('A', end='')
                else:  # Camino
                    print('·', end='')
            else:
                print(simbolos[cara_motor[i, j]], end='')
        print()


# Crear una cara de motor aleatoria
tamano = 10
inicio = (np.random.randint(0, tamano), np.random.randint(0, tamano))  # Punto de inicio, asegurarse que sea un espacio libre
cara_motor = crear_cara_motor(tamano=tamano, probabilidad_obstaculo=0.3, semilla=42, inicio=inicio)

# Definir el objetivo
objetivo = (5, 5)

print("=== Cara del motor original ===")
imprimir_cara_motor(cara_motor)
print("\n")
print("=== Posicion Inicial ===")
print(inicio)
print("=== Posicion objetivo ===")
print(objetivo)
# Ejecutar el algoritmo A*
print("=== Iniciando búsqueda A* ===")
camino = busqueda_a_estrella(cara_motor, inicio, objetivo)

print("\n=== Resultado ===")
if camino:
    print(f"¡Camino encontrado con {len(camino)} pasos!")
    imprimir_cara_motor(cara_motor, camino)
    
    print("\nSecuencia del camino:")
    for i, pos in enumerate(camino):
        print(f"B{i+1}: {pos}")
else:
    print("No se encontró un camino al objetivo.")
