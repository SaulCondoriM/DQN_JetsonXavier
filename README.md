# Sistema de Control Inteligente DQN Distribuido
## Informe Técnico - PC (Simulador) + Jetson Xavier (Agente de IA)

### Resumen Ejecutivo

Este proyecto implementa un **sistema de control inteligente distribuido** basado en **Deep Q-Network (DQN)** para el control autónomo de un robot móvil diferencial. El sistema utiliza una arquitectura cliente-servidor donde:

- **PC (Cliente)**: Ejecuta la simulación del entorno y el robot en Python con visualización en tiempo real
- **Jetson Xavier (Servidor)**: Ejecuta el agente de inteligencia artificial DQN en C++/CUDA para máximo rendimiento

La comunicación entre ambos sistemas se realiza mediante **protocolo TCP/IP**, permitiendo entrenamiento distribuido con separación de responsabilidades: simulación vs. procesamiento de IA.

### Arquitectura del Sistema

```
┌─────────────────────────────────────────┐         TCP/IP          ┌──────────────────────────────────────────┐
│                PC (Cliente)             │ ◄─────────────────────► │         Jetson Xavier (Servidor)         │
│                                         │                         │                                          │
│  ┌─────────────────────────────────┐   │   Estados (10 valores)  │  ┌─────────────────────────────────┐   │
│  │        Simulador 2D             │───┼──────────────────────►  │  │         Servidor TCP            │   │
│  │  • Física del robot             │   │        Formato CSV       │  │  • Parseo de estados            │   │
│  │  • Detección de colisiones      │   │                         │  │  • Validación de datos          │   │
│  │  • Sensores de distancia        │   │   Acciones (0-4)        │  └──────────┬──────────────────────┘   │
│  │  • Cálculo de recompensas       │◄──┼──────────────────────── │              │                          │
│  │  • Gestión de episodios         │   │      (Enteros)          │  ┌───────────▼──────────────────────┐   │
│  └─────────────────────────────────┘   │                         │  │         DQN Agent               │   │
│                                         │                         │  │  • Red Neuronal (CUDA)         │   │
│  ┌─────────────────────────────────┐   │                         │  │  • Replay Buffer                │   │
│  │      Visualización GUI          │   │                         │  │  • Entrenamiento online         │   │
│  │  • Pygame 2D                    │   │                         │  │  • Target Network               │   │
│  │  • Robot, obstáculos, goal      │   │                         │  │  • Política ε-greedy            │   │
│  │  • Trayectorias                 │   │                         │  └─────────────────────────────────┘   │
│  └─────────────────────────────────┘   │                         │                                          │
│                                         │                         │           GPU CUDA Cores                │
└─────────────────────────────────────────┘                         └──────────────────────────────────────────┘
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

#### 📁 PC Simulator (Python)

**`robot_simulator.py`** - *Núcleo de la Simulación*
- **Propósito**: Simula un robot diferencial en entorno 2D con física realista
- **Funcionalidades**:
  - Modelo dinámico del robot (velocidad lineal/angular)
  - Sistema de sensores de distancia (3 sensores: frontal, izquierdo, derecho)
  - Detección de colisiones con obstáculos y límites del entorno
  - Cálculo de recompensas basado en progreso hacia objetivo
  - Gestión de episodios (reset, terminación)
- **Entorno**: 3×3 metros con 2 obstáculos estratégicos
- **Estados**: Vector de 10 dimensiones (posición, orientación, velocidades, sensores, objetivo)

**`visualizer.py`** - *Interfaz Gráfica*
- **Propósito**: Visualización en tiempo real del entrenamiento con Pygame
- **Elementos visuales**:
  - Robot (círculo azul con orientación)
  - Obstáculos (círculos rojos)
  - Objetivo (círculo verde)
  - Trayectoria del robot
  - Información de estado (episodio, recompensa, pasos)
- **Modos**: Visualización continua o por intervalos para optimización

**`tcp_client.py`** - *Comunicación TCP*
- **Propósito**: Cliente TCP que conecta con el agente DQN en Jetson
- **Protocolo**: Envío síncrono de estados y recepción de acciones
- **Manejo de errores**: Reconexión automática y sincronización robusta
- **Formato de datos**: CSV para estados, enteros para acciones

#### 🔧 Jetson Agent (C++/CUDA)

**`neural_network.cuh`** - *Red Neuronal DQN*
- **Propósito**: Implementación de red neuronal profunda en CUDA
- **Arquitectura**: 
  ```
  Input(10) → Dense(128,ReLU) → Dense(128,ReLU) → Dense(64,ReLU) → Output(5)
  ```
- **Optimización**: Kernels CUDA personalizados para forward/backward pass
- **Memoria**: Gestión eficiente de memoria GPU
- **Target Network**: Implementación de red objetivo para estabilidad

**`replay_buffer.hpp`** - *Buffer de Experiencias*
- **Propósito**: Almacenar experiencias (s,a,r,s') para entrenamiento por lotes
- **Capacidad**: 100,000 experiencias con sobrescritura circular
- **Muestreo**: Selección aleatoria de mini-lotes de 64 experiencias
- **Optimización**: Acceso eficiente a memoria para entrenamiento continuo

**`dqn_agent.cuh`** - *Agente DQN Principal*
- **Propósito**: Implementa el algoritmo completo de Deep Q-Network
- **Características**:
  - Política ε-greedy con decaimiento (1.0 → 0.05)
  - Entrenamiento online cada paso
  - Soft update de target network (τ = 0.005)
  - Guardado/carga de modelos entrenados
- **Hiperparámetros optimizados**: lr=0.0005, γ=0.99, batch=64

**`tcp_server.hpp`** - *Servidor TCP Robusto*
- **Propósito**: Servidor TCP que recibe estados y envía acciones
- **Robustez**: Parseo seguro con manejo de errores de formato
- **Funciones de seguridad**: `safe_stof()`, `safe_stoi()` para evitar excepciones
- **Gestión de conexiones**: Manejo de desconexiones y reconexiones

**`main.cu`** - *Programa Principal*
- **Propósito**: Orquestación del entrenamiento completo
- **Loop principal**:
  1. Inicialización de CUDA y red neuronal
  2. Creación de servidor TCP en puerto 5555
  3. Ciclo de entrenamiento episódico
  4. Guardado periódico de modelos
- **Métricas**: Logging de recompensas, epsilon, loss durante entrenamiento

## Protocolo de Comunicación TCP/IP

### Arquitectura de Red

El sistema utiliza un **protocolo TCP personalizado** para comunicación distribuida:

- **Jetson Xavier**: Actúa como **SERVIDOR** (puerto 5555)
- **PC**: Actúa como **CLIENTE** (conecta al Jetson)
- **Protocolo**: Síncrono, un mensaje por paso de simulación

### Datos Enviados: PC → Jetson (Estado del Robot)

El PC envía el estado completo del robot en **formato CSV** con 10 valores:

```csv
x,y,theta,v,omega,d_front,d_left,d_right,dx_goal,dy_goal|done,reward,goal,collision
```

| Campo | Tipo | Rango | Descripción |
|-------|------|-------|-------------|
| `x` | float | [0.0, 3.0] | Posición X del robot (metros) |
| `y` | float | [0.0, 3.0] | Posición Y del robot (metros) |
| `theta` | float | [-π, π] | Orientación del robot (radianes) |
| `v` | float | [0.0, 2.0] | Velocidad lineal actual (m/s) |
| `omega` | float | [-2.0, 2.0] | Velocidad angular actual (rad/s) |
| `d_front` | float | [0.0, 5.0] | Distancia sensor frontal (metros) |
| `d_left` | float | [0.0, 5.0] | Distancia sensor izquierdo (metros) |
| `d_right` | float | [0.0, 5.0] | Distancia sensor derecho (metros) |
| `dx_goal` | float | [-3.0, 3.0] | Componente X hacia objetivo |
| `dy_goal` | float | [-3.0, 3.0] | Componente Y hacia objetivo |

**Ejemplo de mensaje real:**
```
1.5,0.8,0.785,1.2,-0.3,2.1,1.8,3.2,-1.0,1.7|0,-0.1,0,0
```

### Datos Recibidos: Jetson → PC (Acción Seleccionada)

El Jetson responde con un **entero** que representa la acción a ejecutar:

| Acción | Valor | Efecto en el Robot |
|--------|-------|-------------------|
| **FORWARD** | `0` | Acelerar hacia adelante (v += 0.5) |
| **LEFT** | `1` | Girar a la izquierda (ω += 1.0) |
| **RIGHT** | `2` | Girar a la derecha (ω -= 1.0) |
| **BRAKE** | `3` | Frenar (v *= 0.5, ω *= 0.5) |
| **BACKWARD** | `4` | Retroceder (v -= 0.3) |

### Procesamiento en Jetson Xavier

#### 1. **Recepción y Parseo de Estados**
```cpp
// tcp_server.hpp - Parseo robusto
std::vector<float> parseState(const std::string& message) {
    // Separar por comas y convertir con safe_stof()
    // Validar rangos y detectar errores de formato
    // Retornar vector normalizado para la red neuronal
}
```

#### 2. **Inferencia DQN (Forward Pass)**
```cpp
// neural_network.cuh - Procesamiento CUDA
__global__ void forward_pass_kernel(float* input, float* output, 
                                   float* weights, int batch_size) {
    // 1. Propagación hacia adelante en GPU
    // 2. Activaciones ReLU entre capas
    // 3. Cálculo de Q-values para las 5 acciones
}
```

**Flujo de procesamiento:**
1. **Normalización**: Estados se normalizan a rango [0,1]
2. **GPU Transfer**: Datos se copian a memoria GPU
3. **Forward Pass**: Red neuronal procesa entrada → Q-values
4. **Selección de Acción**: ε-greedy sobre Q-values máximos
5. **CPU Return**: Acción seleccionada regresa a CPU

#### 3. **Entrenamiento DQN (Backward Pass)**
```cpp
// dqn_agent.cuh - Entrenamiento online
void train_step() {
    // 1. Muestrear mini-lote del replay buffer (64 experiencias)
    // 2. Calcular Q-targets usando target network
    // 3. Forward pass en main network
    // 4. Calcular loss MSE: (Q_pred - Q_target)²
    // 5. Backward pass y actualización de pesos (Adam optimizer)
    // 6. Soft update de target network
}
```

### Gestión de Errores y Robustez

#### En el PC (Cliente):
- **Reconexión automática** si se pierde conexión
- **Timeout** de 5 segundos por mensaje
- **Validación** de respuestas del Jetson
- **Sincronización** robusta entre episodios

#### En el Jetson (Servidor):
- **Parseo seguro** con funciones `safe_stof()`
- **Validación de rangos** de los estados recibidos
- **Manejo de clientes múltiples** (aunque solo uno activo)
- **Recovery** automático de errores de formato

## Algoritmo Deep Q-Network (DQN)

### Fundamentos Teóricos

El **Deep Q-Network** es un algoritmo de aprendizaje por refuerzo que combina:
- **Q-Learning**: Algoritmo de diferencias temporales para estimar valores Q(s,a)
- **Redes Neuronales Profundas**: Aproximación de funciones para espacios de estados continuos
- **Experience Replay**: Buffer de experiencias para entrenamiento estable
- **Target Network**: Red objetivo para estabilizar el entrenamiento

### Implementación en CUDA

#### **Arquitectura de Red Neuronal**
```
Entrada (10 dimensiones)
    ↓
Capa Densa 1: 10 → 128 neuronas + ReLU
    ↓  
Capa Densa 2: 128 → 128 neuronas + ReLU
    ↓
Capa Densa 3: 128 → 64 neuronas + ReLU  
    ↓
Salida: 64 → 5 Q-values (una por acción)
```

**Parámetros totales**: ~35,000 pesos entrenables

#### **Función de Pérdida (Loss Function)**
```
L(θ) = E[(Q_target - Q_pred)²]

Donde:
Q_target = r + γ * max_a' Q_target(s', a')
Q_pred = Q_main(s, a)
```

#### **Hiperparámetros Optimizados**

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| **Learning Rate** | 0.0005 | Convergencia estable sin overshooting |
| **Gamma (γ)** | 0.99 | Prioriza recompensas futuras (visión a largo plazo) |
| **Epsilon inicial** | 1.0 | Exploración máxima al inicio |
| **Epsilon final** | 0.05 | Mantiene 5% exploración para adaptabilidad |
| **Epsilon decay** | 0.9999 | Decaimiento gradual (5000 episodios) |
| **Batch Size** | 64 | Balance entre estabilidad y eficiencia GPU |
| **Replay Buffer** | 100,000 | Suficiente diversidad sin consumir memoria |
| **Tau (τ)** | 0.005 | Soft update lento para estabilidad |
| **Train Frequency** | 1 | Entrenamiento en cada paso (online) |

### Función de Recompensa Diseñada

La función de recompensa está **cuidadosamente diseñada** para guiar el aprendizaje:

```python
def calculate_reward(self):
    reward = 0.0
    
    # 🎯 OBJETIVO PRINCIPAL
    if self.check_goal_reached():
        return +100.0  # Recompensa máxima por éxito
    
    # ⚠️ PENALIZACIÓN POR COLISIÓN  
    if self.check_collision():
        return -100.0  # Penalización máxima por fallo
    
    # 📈 PROGRESO HACIA EL OBJETIVO
    dist_atual = np.linalg.norm(self.position - self.goal)
    if dist_atual < self.prev_distance:
        reward += 10.0 * (self.prev_distance - dist_atual)  # Recompensa por acercarse
    
    # ⏱️ COSTO POR TIEMPO
    reward -= 0.1  # Incentiva soluciones rápidas
    
    # 🚫 PENALIZACIÓN POR ACCIONES INEFICIENTES
    if action == BRAKE:
        reward -= 0.2  # Desincentivar frenado excesivo
    elif action == BACKWARD:
        reward -= 0.15  # Desincentivar retroceso
    
    # 🎯 BONIFICACIÓN POR PROXIMIDAD AL OBJETIVO
    if dist_atual < 1.0:
        reward += 0.5  # Cerca del objetivo
    elif dist_atual < 2.0:
        reward += 0.2  # Moderadamente cerca
        
    # ⚠️ PENALIZACIÓN POR PROXIMIDAD A OBSTÁCULOS
    if min(d_front, d_left, d_right) < 0.3:
        reward -= 0.5  # Incentiva mantener distancia segura
        
    return reward
```

### Proceso de Entrenamiento

#### **Ciclo de Entrenamiento por Episodio:**

1. **Inicialización**
   - Robot se posiciona en (0.1, 0.1)
   - Objetivo en (2.5, 2.5) 
   - Obstáculos fijos en posiciones estratégicas

2. **Loop de Pasos** (máximo 150 pasos por episodio)
   ```cpp
   for (int step = 0; step < max_steps; step++) {
       // 1. Recibir estado del PC
       state = tcp_server.receive_state();
       
       // 2. Seleccionar acción (ε-greedy)
       action = agent.select_action(state, epsilon);
       
       // 3. Enviar acción al PC
       tcp_server.send_action(action);
       
       // 4. Almacenar experiencia
       replay_buffer.add(prev_state, action, reward, state, done);
       
       // 5. Entrenar si hay suficientes experiencias
       if (replay_buffer.size() >= min_replay_size) {
           agent.train_step();
       }
   }
   ```

3. **Actualización de Parámetros**
   - Decaimiento de epsilon: `ε = ε * decay_rate`
   - Soft update de target network: `θ_target = τ*θ_main + (1-τ)*θ_target`
   - Guardado de modelo cada 100 episodios

#### **Métricas de Entrenamiento Monitoreadas:**

- **Recompensa acumulada por episodio**
- **Tasa de éxito** (episodios que alcanzan objetivo)
- **Número de pasos promedio** hasta completar tarea
- **Loss de la red neuronal** (MSE)
- **Valor de epsilon actual** (exploración vs explotación)

## Configuración de Red y Ejecución

### Configuración de Red Distribuida

#### **Paso 1: Configurar IPs de los Dispositivos**

**En el Jetson Xavier (Servidor):**
```bash
# Obtener IP del Jetson
ip addr show eth0        # Ethernet (recomendado para estabilidad)
# O para WiFi:
ip addr show wlan0

# Ejemplo de salida: inet 192.168.18.114/24
```

**En el PC (Cliente):**
```bash
# Verificar conectividad con el Jetson
ping 192.168.18.114     # Usar la IP real del Jetson

# Opcional: Verificar puerto abierto
nc -zv 192.168.18.114 5555
```

#### **Paso 2: Configurar Firewall (si es necesario)**

**En el Jetson Xavier:**
```bash
# Permitir puerto 5555 para comunicación TCP
sudo ufw allow 5555/tcp

# O desactivar firewall temporalmente durante desarrollo
sudo ufw disable
```

### Instalación y Compilación

#### **En el PC (Python 3.13+)**
```bash
# Instalar dependencias del simulador
pip3 install numpy pygame

# Verificar instalación
cd pc_simulator
python3 -c "import numpy, pygame; print('Dependencias OK')"

# Ejecutar pruebas del simulador
python3 run_tests.py --test simulator
```

#### **En el Jetson Xavier (CUDA 12.2+)**
```bash
# Transferir código al Jetson
scp -r jetson_agent/ usuario@192.168.18.114:~/Final_DQN/

# Conectar al Jetson y compilar
ssh usuario@192.168.18.114
cd ~/Final_DQN/jetson_agent

# Verificar CUDA disponible
nvidia-smi
nvcc --version

# Compilar el agente DQN
make clean && make

# Verificar compilación exitosa
ls -la bin/dqn_agent    # Debe existir el ejecutable
```

### Ejecución del Sistema Distribuido

#### **Secuencia de Inicio (IMPORTANTE: Orden específico)**

**1. Iniciar Servidor DQN en Jetson (PRIMERO):**
```bash
# En terminal del Jetson Xavier
cd ~/Final_DQN/jetson_agent
./bin/dqn_agent --port 5555 --episodes 500

# Salida esperada:
# [CUDA] Inicializando dispositivo GPU...
# [DQN] Creando red neuronal 10->128->128->64->5
# [TCP] Servidor esperando conexiones en puerto 5555...
```

**2. Iniciar Cliente Simulador en PC (SEGUNDO):**
```bash
# En terminal del PC
cd /home/saul/Documentos/Final_DQN/pc_simulator
python3 tcp_client.py --ip 192.168.18.114 --port 5555 --episodes 500 --visualize

# Salida esperada:
# [TCP] Conectando a 192.168.18.114:5555...
# [SIMULATOR] Iniciando entrenamiento DQN...
# [EPISODE 1] Recompensa: -45.2, Pasos: 150, Epsilon: 0.998
```

### Parámetros de Configuración

#### **Opciones del Simulador (PC)**
| Parámetro | Descripción | Valor Default | Rango |
|-----------|-------------|---------------|-------|
| `--ip` | IP del Jetson Xavier | `127.0.0.1` | IP válida |
| `--port` | Puerto TCP de comunicación | `5555` | 1024-65535 |
| `--episodes` | Número total de episodios | `1000` | 1-∞ |
| `--visualize` | Mostrar GUI en tiempo real | `False` | True/False |
| `--render-every` | Renderizar cada N episodios | `10` | 1-100 |
| `--save-logs` | Guardar métricas en archivo | `True` | True/False |

#### **Opciones del Agente DQN (Jetson)**
| Parámetro | Descripción | Valor Default | Rango |
|-----------|-------------|---------------|-------|
| `--port` | Puerto TCP del servidor | `5555` | 1024-65535 |
| `--episodes` | Episodios máximos (-1=∞) | `-1` | -1,1-∞ |
| `--model-path` | Ruta del modelo DQN | `models/dqn_model.bin` | Path válido |
| `--load-model` | Cargar modelo existente | `False` | True/False |
| `--save-every` | Guardar modelo cada N episodios | `100` | 10-1000 |
| `--device-id` | ID del dispositivo CUDA | `0` | 0-N |

### Monitoreo del Entrenamiento

#### **En el Jetson (Logs de IA):**
```
[EPISODE 0001] Reward: -85.4  | Loss: 2.45  | Epsilon: 0.999 | Steps: 150
[EPISODE 0050] Reward: -12.3  | Loss: 0.87  | Epsilon: 0.951 | Steps: 89
[EPISODE 0100] Reward: +45.7  | Loss: 0.34  | Epsilon: 0.905 | Steps: 67
[EPISODE 0200] Reward: +89.2  | Loss: 0.18  | Epsilon: 0.819 | Steps: 34
```

#### **En el PC (Logs de Simulación):**
```
[SIM] Episodio 1/500 | Goal: NO | Colisión: SÍ | Pasos: 150 | R_total: -85.4
[SIM] Episodio 50/500 | Goal: NO | Colisión: SÍ | Pasos: 89 | R_total: -12.3  
[SIM] Episodio 100/500 | Goal: SÍ | Colisión: NO | Pasos: 67 | R_total: +45.7
[SIM] Episodio 200/500 | Goal: SÍ | Colisión: NO | Pasos: 34 | R_total: +89.2
```

## Resultados Esperados y Análisis de Rendimiento

### Curva de Aprendizaje Esperada

El entrenamiento del DQN sigue un patrón característico dividido en **4 fases**:

#### **Fase 1: Exploración Inicial (Episodios 1-50)**
- **Recompensa promedio**: -80 a -50
- **Tasa de éxito**: 0-5%
- **Comportamiento**: Movimientos aleatorios, muchas colisiones
- **Epsilon**: 1.0 → 0.95 (95% exploración)

#### **Fase 2: Aprendizaje Básico (Episodios 51-150)**  
- **Recompensa promedio**: -50 a -10
- **Tasa de éxito**: 5-25%
- **Comportamiento**: Comienza a evitar obstáculos, movimientos más dirigidos
- **Epsilon**: 0.95 → 0.86 (86% exploración)

#### **Fase 3: Refinamiento (Episodios 151-350)**
- **Recompensa promedio**: -10 a +60
- **Tasa de éxito**: 25-70%
- **Comportamiento**: Encuentra rutas válidas consistentemente
- **Epsilon**: 0.86 → 0.70 (70% exploración)

#### **Fase 4: Convergencia (Episodios 351-500+)**
- **Recompensa promedio**: +60 a +95
- **Tasa de éxito**: 70-95%
- **Comportamiento**: Política casi óptima, rutas eficientes
- **Epsilon**: 0.70 → 0.05 (5% exploración residual)

### Métricas de Evaluación

#### **Métricas Primarias:**
- **Tasa de Éxito**: % de episodios que alcanzan el objetivo
- **Recompensa Acumulada**: Suma de recompensas por episodio
- **Pasos hasta Objetivo**: Eficiencia de las rutas encontradas
- **Tiempo de Convergencia**: Episodios necesarios para política estable

#### **Métricas Técnicas:**
- **Loss de Red Neuronal**: Error MSE entre Q_pred y Q_target
- **Utilización de GPU**: % de uso de CUDA cores durante entrenamiento  
- **Throughput**: Pasos procesados por segundo
- **Memoria GPU**: Uso de VRAM para redes y replay buffer

## Pruebas y Validación

### Configuración del Entorno de Prueba

#### **Entorno Optimizado para Aprendizaje Rápido:**
- **Dimensiones**: 3×3 metros (reducido para acelerar convergencia)
- **Posición inicial robot**: (0.1, 0.1) 
- **Objetivo**: (2.5, 2.5)
- **Obstáculos**: 2 obstáculos estratégicamente ubicados
  - Obstáculo 1: Centro (1.5, 1.5), radio 0.35m - bloquea ruta directa
  - Obstáculo 2: (2.2, 1.0), radio 0.2m - bloquea diagonal inferior
- **Pasos máximos**: 150 por episodio

#### **Validación de Dificultad del Entorno:**
Antes del entrenamiento DQN, se validó que el entorno requiere aprendizaje:

```bash
# Prueba con política directa (sin evasión)
=== Test: Política DIRECTA (sin evasión) ===
Resultado: 0/10 goals, 10/10 colisiones

# Prueba con política aleatoria  
=== Test: Política ALEATORIA ===
Resultado: 0/10 goals, 10/10 colisiones

>> Un DQN entrenado debería superar ambas políticas!
```

**Conclusión**: Ambiente desafiante que requiere aprendizaje para tener éxito.

### Pruebas Locales de Desarrollo

Para desarrollo y debugging sin el Jetson, usa el servidor de prueba:

#### **Servidor de Prueba en Python:**
```bash
# Terminal 1: Servidor de prueba (simula Jetson)
cd pc_simulator  
python3 test_server.py --port 5555 --episodes 100 --policy random

# Terminal 2: Cliente simulador
cd pc_simulator
python3 tcp_client.py --ip 127.0.0.1 --port 5555 --episodes 100 --visualize
```

#### **Políticas de Prueba Disponibles:**
| Política | Descripción | Uso |
|----------|-------------|-----|
| `random` | Acciones aleatorias uniformes | Baseline inferior |
| `forward` | Solo avanzar (sin evasión) | Validar obstáculos |
| `simple` | Giros simples al detectar obstáculos | Heurística básica |
| `dqn` | Cargar modelo DQN entrenado | Validar agente |

### Suite de Pruebas Automatizadas

#### **Ejecutar todas las pruebas:**
```bash
cd pc_simulator
python3 run_tests.py --all

# O pruebas específicas:
python3 run_tests.py --test simulator      # Prueba simulador solo  
python3 run_tests.py --test tcp           # Prueba comunicación TCP
python3 run_tests.py --test environment   # Prueba configuración entorno
```

#### **Pruebas de Rendimiento:**
```bash
# Benchmark de throughput del simulador
python3 run_tests.py --test performance --episodes 1000

# Salida esperada:
# [PERF] Simulación: 2847 pasos/segundo
# [PERF] TCP: 1923 mensajes/segundo  
# [PERF] Renderización: 60 FPS promedio
```

## Solución de Problemas y Debugging

### Problemas Comunes de Conectividad

#### **Error: "Connection refused" o "No route to host"**
```bash
# 1. Verificar que el Jetson esté ejecutando el servidor
ssh usuario@192.168.18.114
ps aux | grep dqn_agent    # Debe aparecer el proceso

# 2. Verificar IP correcta del Jetson
ip addr show eth0          # Confirmar IP real

# 3. Probar conectividad básica
ping 192.168.18.114        # Desde el PC
nc -zv 192.168.18.114 5555 # Probar puerto específico

# 4. Configurar firewall en Jetson
sudo ufw allow 5555/tcp
# O temporalmente: sudo ufw disable
```

#### **Error: "CUDA out of memory"**
```bash
# 1. Verificar memoria GPU disponible
nvidia-smi

# 2. Si hay otros procesos usando GPU, terminarlos
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID_DEL_PROCESO>

# 3. Reducir batch size en main.cu si es necesario
# Cambiar BATCH_SIZE de 64 a 32 o 16
```

#### **Error: "stof exception" o parseo de datos**
```bash
# Verificar formato de mensajes TCP
# En pc_simulator/tcp_client.py, añadir debug:
print(f"Enviando: {message}")

# En jetson_agent/include/tcp_server.hpp
# Ya tiene manejo robusto con safe_stof()
```

### Problemas de Entrenamiento

#### **El agente no aprende (recompensa no mejora)**
1. **Verificar replay buffer**: Debe acumular al menos 500 experiencias
2. **Ajustar epsilon decay**: Muy rápido impide exploración
3. **Revisar función de recompensa**: Debe dar feedback útil
4. **Aumentar episodios**: DQN necesita 200-500 episodios mínimo

#### **Convergencia muy lenta**  
1. **Reducir tamaño del entorno**: Ya optimizado a 3×3m
2. **Ajustar learning rate**: Probar 0.001 si 0.0005 es muy lento
3. **Simplificar obstáculos**: Reducir de 2 a 1 obstáculo temporalmente

#### **Inestabilidad en el entrenamiento**
1. **Verificar target network**: Debe actualizarse cada 100 pasos
2. **Revisar soft update tau**: 0.005 es conservativo y estable
3. **Monitorear loss**: No debe crecer indefinidamente

### Debugging Avanzado

#### **Logs Detallados en Jetson:**
```cpp
// En main.cu, añadir:
#define DEBUG_MODE 1

// Habilita logs extendidos:
// [DEBUG] Estado recibido: [1.2, 0.8, 0.785, ...]
// [DEBUG] Q-values: [0.23, -0.45, 0.78, -0.12, 0.34]
// [DEBUG] Acción seleccionada: 2 (epsilon=0.891)
```

#### **Profiling de Rendimiento:**
```bash
# En Jetson, usar nvprof para análisis CUDA
nvprof --log-file dqn_profile.txt ./bin/dqn_agent --episodes 10

# Analizar cuellos de botella:
# GPU utilization, memory transfers, kernel execution time
```

#### **Visualización de Métricas:**
```python
# En PC, modificar tcp_client.py para guardar métricas
import matplotlib.pyplot as plt

rewards = []  # Recolectar durante entrenamiento
plt.plot(rewards)
plt.xlabel('Episodio')
plt.ylabel('Recompensa Acumulada')  
plt.title('Curva de Aprendizaje DQN')
plt.show()
```

## Notas Técnicas sobre Hardware

### Requisitos del Sistema

#### **PC (Cliente - Simulador):**
- **CPU**: Intel/AMD multi-core (4+ cores recomendado)
- **RAM**: 4 GB mínimo, 8 GB recomendado  
- **Python**: 3.8+ (probado con Python 3.13)
- **Dependencias**: NumPy, Pygame
- **Red**: Conexión Ethernet/WiFi estable con Jetson

#### **Jetson Xavier (Servidor - Agente IA):**
- **GPU**: 512 CUDA cores (Volta), 32 Tensor cores
- **RAM**: 16/32 GB (probado con 32 GB)
- **CUDA**: Compute Capability 7.2 (sm_72)
- **Storage**: 10 GB disponible para modelos y logs
- **Red**: Ethernet preferido para estabilidad

### Configuración CUDA por Plataforma

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

### Demo Interactivo del Simulador

Para explorar manualmente el entorno antes del entrenamiento:

```bash
cd pc_simulator
python3 visualizer.py
```

#### **Controles de Teclado:**
| Tecla | Acción | Efecto en Robot |
|-------|--------|-----------------|
| **W** | Avanzar | v += 0.5 m/s |
| **S** | Frenar | v *= 0.5, ω *= 0.5 |
| **A** | Girar izquierda | ω += 1.0 rad/s |
| **D** | Girar derecha | ω -= 1.0 rad/s |
| **X** | Retroceder | v -= 0.3 m/s |
| **R** | Reiniciar | Nueva posición aleatoria |
| **Q** | Salir | Cerrar simulador |

**Objetivo del demo**: Entender la dinámica del robot y la dificultad del entorno antes de entrenar la IA.

### Optimizaciones de Rendimiento Implementadas

#### **En el Simulador (Python):**
- **Vectorización NumPy**: Cálculos de sensores y física optimizados
- **Renderizado condicional**: Solo renderiza cuando es necesario
- **TCP sin bloqueo**: Timeouts para evitar cuelgues
- **Cache de colisiones**: Evita recálculos innecesarios

#### **En el Agente (CUDA):**
- **Memory coalescing**: Accesos alineados a memoria GPU
- **Shared memory**: Cache local para pesos de red neuronal  
- **Kernels fusionados**: Forward+backward pass en un solo kernel
- **Streams CUDA**: Paralelización de transfers CPU↔GPU
- **Target network soft update**: Actualización eficiente en GPU

### Extensiones Futuras Posibles

#### **Mejoras del Entorno:**
- Entornos dinámicos con obstáculos móviles
- Múltiples robots colaborativos
- Objetivos múltiples o secuenciales
- Ruido en sensores para realismo

#### **Mejoras del Algoritmo:**
- Dueling DQN para mejor estimación de valores
- Prioritized Experience Replay para muestras importantes
- Rainbow DQN con todas las mejoras combinadas
- Multi-agent Deep Q-Network (MADQN)

#### **Integración con Robot Real:**
- ROS 2 para interfaz con hardware real
- Cámara RGB-D para sensores visuales
- LIDAR para navegación precisa
- Actuadores servo para control de motores

---

## Conclusiones

Este sistema demuestra la **implementación exitosa de un DQN distribuido** con separación clara de responsabilidades:

- **PC**: Se enfoca en simulación realista y visualización
- **Jetson**: Maximiza el rendimiento de IA con CUDA
- **TCP**: Permite escalabilidad y flexibilidad de despliegue

La arquitectura es **extensible y modular**, facilitando mejoras futuras tanto en algoritmos de IA como en complejidad del entorno de simulación.

**Desarrollado en Diciembre 2025** - Proyecto de Control Inteligente con Deep Reinforcement Learning
