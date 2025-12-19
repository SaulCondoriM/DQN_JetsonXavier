# Sistema de Control Inteligente DQN Distribuido PC (Simulador) + Jetson Xavier (Agente de IA)

### Integrantes del Proyecto

| Nombre | Aporte |
|--------|--------|
| Saul Condori Machaca | 22.0 |
| Christian Pardave Espinoza | 19.5 |
| Merisable Ruelas Quenaya | 19.5 |
| Yanira Suni Quispe | 19.5 |
| Katherine Bejar Roman | 19.5 |

### Resumen

Este proyecto implementa un sistema de control inteligente distribuido basado en Deep Q-Network (DQN) para el control autónomo de un robot móvil diferencial. El sistema utiliza una arquitectura cliente-servidor donde:

- **PC (Cliente)**: Ejecuta la simulación del entorno y el robot en Python con visualización en tiempo real
- **Jetson Xavier (Servidor)**: Ejecuta el agente de inteligencia artificial DQN en C++/CUDA para máximo rendimiento

La comunicación entre ambos sistemas se realiza mediante protocolo TCP/IP, permitiendo entrenamiento distribuido con separación de responsabilidades: simulación vs. procesamiento de IA.

### Arquitectura del Sistema

```
PC (Cliente)                         TCP/IP                    Jetson Xavier (Servidor)
                                                              
Simulador 2D            ------>   Estados (10 valores)  ------>   Servidor TCP
- Física del robot                  Formato CSV                  - Parseo de estados
- Detección colisiones                                           - Validación de datos
- Sensores distancia                                                     |
- Cálculo recompensas   <------    Acciones (0-4)      <------    DQN Agent
- Gestión episodios                  (Enteros)                   - Red Neuronal (CUDA)
                                                                 - Replay Buffer
Visualización GUI                                                - Entrenamiento online
- Pygame 2D                                                      - Target Network
- Robot, obstáculos, goal                                        - Política epsilon-greedy
- Trayectorias                                                   
                                                                 GPU CUDA Cores
```

## Estructura del Proyecto y Descripción de Archivos

### Organización del Código

```
Final_DQN/
├── pc_simulator/           # 🖥️  Código PC (Python) - Simulación y Visualización
│   ├── robot_simulator.py  # Simulador principal del robot diferencial
│   ├── visualizer.py       # Interfaz gráfica con Pygame
│   ├── tcp_client.py       # Cliente TCP de comunicación
│   ├── test_server.py      # Servidor de pruebas para desarrollo local
│   └── run_tests.py        # Suite de pruebas automatizadas
│
├── jetson_agent/           # 🚀 Código Jetson (C++/CUDA) - Inteligencia Artificial  
│   ├── include/            # Headers de las clases principales
│   │   ├── cuda_utils.cuh     # Utilidades y macros CUDA
│   │   ├── neural_network.cuh # Implementación de red neuronal DQN
│   │   ├── replay_buffer.hpp  # Buffer de experiencias para entrenamiento
│   │   ├── dqn_agent.cuh      # Agente DQN completo con algoritmo
│   │   └── tcp_server.hpp     # Servidor TCP robusto con manejo de errores
│   ├── src/
│   │   └── main.cu           # Programa principal y loop de entrenamiento
│   └── Makefile              # Compilación optimizada para CUDA
│
└── README.md                # Este documento
```

### Descripción Detallada de Archivos

#### PC Simulator (Python)

**`robot_simulator.py`** - Núcleo de la Simulación
- Simula un robot diferencial en entorno 2D con física realista
- Modelo dinámico del robot (velocidad lineal/angular)
- Sistema de sensores de distancia (3 sensores: frontal, izquierdo, derecho)
- Detección de colisiones con obstáculos y límites del entorno
- Cálculo de recompensas basado en progreso hacia objetivo
- Gestión de episodios (reset, terminación)
- Entorno: 3x3 metros con 2 obstáculos estratégicos
- Estados: Vector de 10 dimensiones (posición, orientación, velocidades, sensores, objetivo)

**`visualizer.py`** - Interfaz Gráfica
- Visualización en tiempo real del entrenamiento con Pygame
- Elementos visuales: Robot (círculo azul con orientación), Obstáculos (círculos rojos), Objetivo (círculo verde), Trayectoria del robot
- Información de estado (episodio, recompensa, pasos)
- Modos: Visualización continua o por intervalos para optimización

**`tcp_client.py`** - Comunicación TCP
- Cliente TCP que conecta con el agente DQN en Jetson
- Protocolo: Envío síncrono de estados y recepción de acciones
- Formato de datos: CSV para estados, enteros para acciones

#### Jetson Agent (C++/CUDA)

**`neural_network.cuh`** - Red Neuronal DQN
- Implementación de red neuronal profunda en CUDA
- Arquitectura: Input(10) -> Dense(128,ReLU) -> Dense(128,ReLU) -> Dense(64,ReLU) -> Output(5)
- Kernels CUDA personalizados para forward/backward pass
- Gestión eficiente de memoria GPU
- Target Network para estabilidad

**`replay_buffer.hpp`** - Buffer de Experiencias
- Almacenar experiencias (s,a,r,s') para entrenamiento por lotes
- Capacidad: 100,000 experiencias con sobrescritura circular
- Muestreo: Selección aleatoria de mini-lotes de 64 experiencias

**`dqn_agent.cuh`** - Agente DQN Principal
- Implementa el algoritmo completo de Deep Q-Network
- Política epsilon-greedy con decaimiento (1.0 -> 0.05)
- Entrenamiento online cada paso
- Soft update de target network (tau = 0.005)
- Guardado/carga de modelos entrenados
- Hiperparámetros optimizados: lr=0.0005, gamma=0.99, batch=64

**`tcp_server.hpp`** - Servidor TCP
- Servidor TCP que recibe estados y envía acciones
- Parseo seguro con manejo de errores de formato
- Funciones de seguridad: safe_stof(), safe_stoi()

**`main.cu`** - Programa Principal
- Orquestación del entrenamiento completo
- Inicialización de CUDA y red neuronal
- Creación de servidor TCP en puerto 5555
- Ciclo de entrenamiento episódico
- Guardado periódico de modelos
- Logging de recompensas, epsilon, loss durante entrenamiento

## Protocolo de Comunicación TCP/IP

### Arquitectura de Red

- **Jetson Xavier**: Actúa como SERVIDOR (puerto 5555)
- **PC**: Actúa como CLIENTE (conecta al Jetson)
- **Protocolo**: Síncrono, un mensaje por paso de simulación

### Datos Enviados: PC -> Jetson (Estado del Robot)

El PC envía el estado completo del robot en formato CSV con 10 valores:

```csv
x,y,theta,v,omega,d_front,d_left,d_right,dx_goal,dy_goal|done,reward,goal,collision
```

| Campo | Tipo | Rango | Descripción |
|-------|------|-------|-------------|
| `x` | float | [0.0, 3.0] | Posición X del robot (metros) |
| `y` | float | [0.0, 3.0] | Posición Y del robot (metros) |
| `theta` | float | [-pi, pi] | Orientación del robot (radianes) |
| `v` | float | [0.0, 2.0] | Velocidad lineal actual (m/s) |
| `omega` | float | [-2.0, 2.0] | Velocidad angular actual (rad/s) |
| `d_front` | float | [0.0, 5.0] | Distancia sensor frontal (metros) |
| `d_left` | float | [0.0, 5.0] | Distancia sensor izquierdo (metros) |
| `d_right` | float | [0.0, 5.0] | Distancia sensor derecho (metros) |
| `dx_goal` | float | [-3.0, 3.0] | Componente X hacia objetivo |
| `dy_goal` | float | [-3.0, 3.0] | Componente Y hacia objetivo |

Ejemplo de mensaje real:
```
1.5,0.8,0.785,1.2,-0.3,2.1,1.8,3.2,-1.0,1.7|0,-0.1,0,0
```

### Datos Recibidos: Jetson -> PC (Acción Seleccionada)

El Jetson responde con un entero que representa la acción a ejecutar:

| Acción | Valor | Efecto en el Robot |
|--------|-------|-------------------|
| FORWARD | 0 | Acelerar hacia adelante (v += 0.5) |
| LEFT | 1 | Girar a la izquierda (omega += 1.0) |
| RIGHT | 2 | Girar a la derecha (omega -= 1.0) |
| BRAKE | 3 | Frenar (v *= 0.5, omega *= 0.5) |
| BACKWARD | 4 | Retroceder (v -= 0.3) |

### Procesamiento en Jetson Xavier

1. Recepción y Parseo de Estados
   - Separar por comas y convertir con safe_stof()
   - Validar rangos y detectar errores de formato
   - Retornar vector normalizado para la red neuronal

2. Inferencia DQN (Forward Pass)
   - Normalización: Estados se normalizan a rango [0,1]
   - GPU Transfer: Datos se copian a memoria GPU
   - Forward Pass: Red neuronal procesa entrada -> Q-values
   - Selección de Acción: epsilon-greedy sobre Q-values máximos
   - CPU Return: Acción seleccionada regresa a CPU

3. Entrenamiento DQN (Backward Pass)
   - Muestrear mini-lote del replay buffer (64 experiencias)
   - Calcular Q-targets usando target network
   - Forward pass en main network
   - Calcular loss MSE: (Q_pred - Q_target)^2
   - Backward pass y actualización de pesos (Adam optimizer)
   - Soft update de target network

## Algoritmo Deep Q-Network (DQN)

### Arquitectura de Red Neuronal

```
Entrada (10 dimensiones)
    |
Capa Densa 1: 10 -> 128 neuronas + ReLU
    |  
Capa Densa 2: 128 -> 128 neuronas + ReLU
    |
Capa Densa 3: 128 -> 64 neuronas + ReLU  
    |
Salida: 64 -> 5 Q-values (una por acción)
```

Parámetros totales: ~35,000 pesos entrenables

### Función de Pérdida (Loss Function)

```
L(theta) = E[(Q_target - Q_pred)^2]

Donde:
Q_target = r + gamma * max_a' Q_target(s', a')
Q_pred = Q_main(s, a)
```

### Hiperparámetros

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Learning Rate | 0.0005 | Tasa de aprendizaje |
| Gamma | 0.99 | Factor de descuento |
| Epsilon inicial | 1.0 | Exploración inicial |
| Epsilon final | 0.05 | Exploración mínima |
| Epsilon decay | 0.9999 | Decaimiento de exploración |
| Batch Size | 64 | Tamaño del mini-lote |
| Replay Buffer | 100,000 | Capacidad del buffer |
| Tau | 0.005 | Factor de actualización de target network |
| Train Frequency | 1 | Entrenamiento en cada paso |

### Función de Recompensa

```python
def calculate_reward(self):
    reward = 0.0
    
    # Objetivo alcanzado
    if self.check_goal_reached():
        return +100.0
    
    # Colisión
    if self.check_collision():
        return -100.0
    
    # Progreso hacia el objetivo
    dist_actual = np.linalg.norm(self.position - self.goal)
    if dist_actual < self.prev_distance:
        reward += 10.0 * (self.prev_distance - dist_actual)
    
    # Costo por tiempo
    reward -= 0.1
    
    # Penalización por acciones ineficientes
    if action == BRAKE:
        reward -= 0.2
    elif action == BACKWARD:
        reward -= 0.15
    
    # Bonificación por proximidad al objetivo
    if dist_actual < 1.0:
        reward += 0.5
    elif dist_actual < 2.0:
        reward += 0.2
        
    # Penalización por proximidad a obstáculos
    if min(d_front, d_left, d_right) < 0.3:
        reward -= 0.5
        
    return reward
```

### Proceso de Entrenamiento

#### Ciclo de Entrenamiento por Episodio:

1. Inicialización
   - Robot en posición (0.1, 0.1)
   - Objetivo en (2.5, 2.5) 
   - Obstáculos fijos en posiciones estratégicas

2. Loop de Pasos (máximo 150 pasos por episodio)
   - Recibir estado del PC
   - Seleccionar acción (epsilon-greedy)
   - Enviar acción al PC
   - Almacenar experiencia
   - Entrenar si hay suficientes experiencias

3. Actualización de Parámetros
   - Decaimiento de epsilon
   - Soft update de target network
   - Guardado de modelo cada 100 episodios

## Configuración y Ejecución

### Configuración de Red

#### Paso 1: Configurar IPs de los Dispositivos

En el Jetson Xavier (Servidor):
```bash
# Obtener IP del Jetson
ip addr show eth0        # Ethernet
# O para WiFi:
ip addr show wlan0

# Ejemplo de salida: inet 192.168.18.114/24
```

En el PC (Cliente):
```bash
# Verificar conectividad con el Jetson
ping 192.168.18.114     # Usar la IP real del Jetson

# Verificar puerto abierto
nc -zv 192.168.18.114 5555
```

#### Paso 2: Configurar Firewall (si es necesario)

En el Jetson Xavier:
```bash
# Permitir puerto 5555 para comunicación TCP
sudo ufw allow 5555/tcp

# O desactivar firewall temporalmente
sudo ufw disable
```

### Instalación y Compilación

#### En el PC (Python 3.8+)
```bash
# Instalar dependencias del simulador
pip3 install numpy pygame

# Verificar instalación
cd pc_simulator
python3 -c "import numpy, pygame; print('Dependencias OK')"
```

#### En el Jetson Xavier (CUDA 12.2+)
```bash
# Transferir código al Jetson
scp -r jetson_agent/ usuario@192.168.18.114:~/DQN_JetsonXavier/

# Conectar al Jetson y compilar
ssh usuario@192.168.18.114
cd ~/DQN_JetsonXavier/jetson_agent

# Verificar CUDA disponible
nvidia-smi
nvcc --version

# Compilar el agente DQN
make clean && make

# Verificar compilación exitosa
ls -la bin/dqn_agent
```

### Ejecución del Sistema

#### Secuencia de Inicio (IMPORTANTE: Orden específico)

1. Iniciar Servidor DQN en Jetson (PRIMERO):
```bash
# En terminal del Jetson Xavier
cd ~/DQN_JetsonXavier/jetson_agent
./bin/dqn_agent --port 5555 --episodes 500
```

2. Iniciar Cliente Simulador en PC (SEGUNDO):
```bash
# En terminal del PC
cd pc_simulator
python3 tcp_client.py --ip 192.168.18.114 --port 5555 --episodes 500 --visualize
```

### Parámetros de Configuración

#### Opciones del Simulador (PC)
| Parámetro | Descripción | Valor Default |
|-----------|-------------|---------------|
| --ip | IP del Jetson Xavier | 127.0.0.1 |
| --port | Puerto TCP de comunicación | 5555 |
| --episodes | Número total de episodios | 1000 |
| --visualize | Mostrar GUI en tiempo real | False |
| --render-every | Renderizar cada N episodios | 10 |
| --save-logs | Guardar métricas en archivo | True |

#### Opciones del Agente DQN (Jetson)
| Parámetro | Descripción | Valor Default |
|-----------|-------------|---------------|
| --port | Puerto TCP del servidor | 5555 |
| --episodes | Episodios máximos (-1=infinito) | -1 |
| --model-path | Ruta del modelo DQN | models/dqn_model.bin |
| --load-model | Cargar modelo existente | False |
| --save-every | Guardar modelo cada N episodios | 100 |
| --device-id | ID del dispositivo CUDA | 0 |

### Monitoreo del Entrenamiento

#### En el Jetson (Logs de IA):
```
[EPISODE 0001] Reward: -85.4  | Loss: 2.45  | Epsilon: 0.999 | Steps: 150
[EPISODE 0050] Reward: -12.3  | Loss: 0.87  | Epsilon: 0.951 | Steps: 89
[EPISODE 0100] Reward: +45.7  | Loss: 0.34  | Epsilon: 0.905 | Steps: 67
[EPISODE 0200] Reward: +89.2  | Loss: 0.18  | Epsilon: 0.819 | Steps: 34
```

#### En el PC (Logs de Simulación):
```
[SIM] Episodio 1/500 | Goal: NO | Colisión: SI | Pasos: 150 | R_total: -85.4
[SIM] Episodio 50/500 | Goal: NO | Colisión: SI | Pasos: 89 | R_total: -12.3  
[SIM] Episodio 100/500 | Goal: SI | Colisión: NO | Pasos: 67 | R_total: +45.7
[SIM] Episodio 200/500 | Goal: SI | Colisión: NO | Pasos: 34 | R_total: +89.2
```

## Requisitos del Sistema

### PC (Cliente - Simulador)
- CPU: Intel/AMD multi-core (4+ cores recomendado)
- RAM: 4 GB mínimo, 8 GB recomendado  
- Python: 3.8+
- Dependencias: NumPy, Pygame
- Red: Conexión Ethernet/WiFi estable con Jetson

### Jetson Xavier (Servidor - Agente IA)
- GPU: 512 CUDA cores (Volta), 32 Tensor cores
- RAM: 16/32 GB
- CUDA: Compute Capability 7.2 (sm_72)
- Storage: 10 GB disponible para modelos y logs
- Red: Ethernet preferido para estabilidad

### Configuración CUDA

El Makefile incluye optimizaciones específicas por arquitectura:

```makefile
# Jetson Nano (Maxwell): sm_53
# NVCC_FLAGS += -arch=sm_53

# Jetson TX2 (Pascal): sm_62  
# NVCC_FLAGS += -arch=sm_62

# Jetson Xavier (Volta): sm_72 [ACTUAL]
NVCC_FLAGS += -arch=sm_72

# Jetson Orin (Ampere): sm_87
# NVCC_FLAGS += -arch=sm_87
```

