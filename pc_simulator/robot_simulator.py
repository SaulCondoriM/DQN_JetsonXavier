"""
Simulador 2D para Robot Diferencial tipo LEGO Mindstorms EV3 Gyro Boy
Este simulador modela:
- Cinemática diferencial
- Obstáculos estáticos
- Sensores de distancia simulados
- Punto objetivo (goal)
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class RobotState:
    """Estado del robot"""
    x: float = 0.0          # Posición X (metros)
    y: float = 0.0          # Posición Y (metros)
    theta: float = 0.0      # Orientación (radianes)
    v: float = 0.0          # Velocidad lineal (m/s)
    omega: float = 0.0      # Velocidad angular (rad/s)


@dataclass
class Obstacle:
    """Obstáculo circular en el entorno"""
    x: float
    y: float
    radius: float


class DifferentialRobotSimulator:
    """
    Simulador de robot diferencial 2D con sensores y obstáculos
    """
    
    # Parámetros del robot (tipo EV3)
    WHEEL_RADIUS = 0.028      # Radio de rueda (28mm)
    WHEEL_BASE = 0.12         # Distancia entre ruedas (12cm)
    ROBOT_RADIUS = 0.08       # Radio del robot para colisiones (8cm)
    
    # Límites de velocidad
    MAX_LINEAR_VEL = 0.3      # m/s
    MAX_ANGULAR_VEL = 2.0     # rad/s
    
    # Parámetros del entorno (MUY PEQUEÑO para aprendizaje rápido)
    ENV_WIDTH = 2.0           # Ancho del entorno (metros)
    ENV_HEIGHT = 2.0          # Alto del entorno (metros)
    
    # Parámetros de sensores
    SENSOR_MAX_RANGE = 1.0    # Rango máximo de sensores (metros)
    NUM_SENSOR_RAYS = 36      # Rayos para detección de obstáculos
    
    # Acciones discretas
    ACTIONS = {
        0: "avanzar",
        1: "girar_izquierda",
        2: "girar_derecha",
        3: "frenar",
        4: "retroceder"
    }
    
    def __init__(self, goal_position: Tuple[float, float] = (1.8, 1.8)):
        """
        Inicializa el simulador
        
        Args:
            goal_position: Posición del objetivo (x, y)
        """
        self.robot = RobotState()
        self.goal = np.array(goal_position)
        self.obstacles: List[Obstacle] = []
        self.dt = 0.05  # Paso de tiempo (50ms)
        self.episode_steps = 0
        self.max_episode_steps = 200  # Muy reducido para entorno 2x2
        self.collision = False
        self.goal_reached = False
        self.goal_threshold = 0.2  # Distancia para considerar meta alcanzada
        
        self._setup_default_obstacles()
        self.reset()
    
    def _setup_default_obstacles(self):
        """Configura obstáculos por defecto en el entorno"""
        # Entorno 2x2: Un obstáculo bloquea parcialmente el camino diagonal
        # El agente debe aprender a rodearlo
        self.obstacles = [
            Obstacle(1.0, 1.0, 0.25),  # Obstáculo central - bloquea ruta directa
            Obstacle(1.5, 0.7, 0.15),  # Obstáculo secundario abajo-derecha
        ]
    
    def reset(self, start_position: Optional[Tuple[float, float]] = None,
              start_theta: Optional[float] = None) -> np.ndarray:
        """
        Reinicia el episodio
        
        Args:
            start_position: Posición inicial opcional
            start_theta: Orientación inicial opcional
            
        Returns:
            Estado inicial del robot
        """
        if start_position is None:
            # Posición inicial en esquina inferior izquierda (entorno 3x3)
            self.robot.x = np.random.uniform(0.2, 0.5)
            self.robot.y = np.random.uniform(0.2, 0.5)
        else:
            self.robot.x, self.robot.y = start_position
            
        if start_theta is None:
            # Orientación inicial hacia el objetivo aproximadamente
            dx = self.goal[0] - self.robot.x
            dy = self.goal[1] - self.robot.y
            self.robot.theta = math.atan2(dy, dx) + np.random.uniform(-0.3, 0.3)
        else:
            self.robot.theta = start_theta
            
        self.robot.v = 0.0
        self.robot.omega = 0.0
        self.episode_steps = 0
        self.collision = False
        self.goal_reached = False
        
        return self.get_state()
    
    def apply_action(self, action: int) -> Tuple[float, float]:
        """
        Convierte acción discreta a comandos de velocidad
        
        Args:
            action: Índice de acción (0-4)
            
        Returns:
            Tupla (velocidad_lineal, velocidad_angular)
        """
        if action == 0:  # Avanzar
            return (self.MAX_LINEAR_VEL, 0.0)
        elif action == 1:  # Girar izquierda
            return (self.MAX_LINEAR_VEL * 0.3, self.MAX_ANGULAR_VEL * 0.5)
        elif action == 2:  # Girar derecha
            return (self.MAX_LINEAR_VEL * 0.3, -self.MAX_ANGULAR_VEL * 0.5)
        elif action == 3:  # Frenar
            return (0.0, 0.0)
        elif action == 4:  # Retroceder
            return (-self.MAX_LINEAR_VEL * 0.5, 0.0)
        else:
            return (0.0, 0.0)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Ejecuta un paso de simulación
        
        Args:
            action: Índice de acción (0-4)
            
        Returns:
            Tupla (nuevo_estado, recompensa, terminado, info)
        """
        # Guardar posición anterior para cálculo de recompensa
        prev_distance_to_goal = self._distance_to_goal()
        
        # Aplicar acción
        v_cmd, omega_cmd = self.apply_action(action)
        
        # Actualizar velocidades con suavizado (simula inercia)
        alpha = 0.7  # Factor de suavizado
        self.robot.v = alpha * v_cmd + (1 - alpha) * self.robot.v
        self.robot.omega = alpha * omega_cmd + (1 - alpha) * self.robot.omega
        
        # Cinemática diferencial
        # dx/dt = v * cos(theta)
        # dy/dt = v * sin(theta)
        # dtheta/dt = omega
        self.robot.x += self.robot.v * math.cos(self.robot.theta) * self.dt
        self.robot.y += self.robot.v * math.sin(self.robot.theta) * self.dt
        self.robot.theta += self.robot.omega * self.dt
        
        # Normalizar ángulo a [-pi, pi]
        self.robot.theta = math.atan2(math.sin(self.robot.theta), 
                                       math.cos(self.robot.theta))
        
        self.episode_steps += 1
        
        # Verificar colisiones
        self.collision = self._check_collision()
        
        # Verificar si llegó al objetivo
        current_distance_to_goal = self._distance_to_goal()
        self.goal_reached = current_distance_to_goal < self.goal_threshold
        
        # Calcular recompensa
        reward = self._calculate_reward(prev_distance_to_goal, 
                                         current_distance_to_goal, 
                                         action)
        
        # Verificar condiciones de terminación
        done = self.collision or self.goal_reached or \
               self.episode_steps >= self.max_episode_steps or \
               self._out_of_bounds()
        
        info = {
            "collision": self.collision,
            "goal_reached": self.goal_reached,
            "steps": self.episode_steps,
            "distance_to_goal": current_distance_to_goal
        }
        
        return self.get_state(), reward, done, info
    
    def _distance_to_goal(self) -> float:
        """Calcula distancia euclidiana al objetivo"""
        return math.sqrt((self.robot.x - self.goal[0])**2 + 
                        (self.robot.y - self.goal[1])**2)
    
    def _check_collision(self) -> bool:
        """Verifica colisión con obstáculos o paredes"""
        # Colisión con paredes
        if (self.robot.x - self.ROBOT_RADIUS < 0 or 
            self.robot.x + self.ROBOT_RADIUS > self.ENV_WIDTH or
            self.robot.y - self.ROBOT_RADIUS < 0 or 
            self.robot.y + self.ROBOT_RADIUS > self.ENV_HEIGHT):
            return True
        
        # Colisión con obstáculos
        for obs in self.obstacles:
            dist = math.sqrt((self.robot.x - obs.x)**2 + 
                           (self.robot.y - obs.y)**2)
            if dist < (self.ROBOT_RADIUS + obs.radius):
                return True
        
        return False
    
    def _out_of_bounds(self) -> bool:
        """Verifica si el robot está fuera de límites"""
        margin = 0.5
        return (self.robot.x < -margin or 
                self.robot.x > self.ENV_WIDTH + margin or
                self.robot.y < -margin or 
                self.robot.y > self.ENV_HEIGHT + margin)
    
    def _calculate_reward(self, prev_dist: float, curr_dist: float, 
                         action: int) -> float:
        """
        Calcula la recompensa del paso actual
        
        Args:
            prev_dist: Distancia anterior al objetivo
            curr_dist: Distancia actual al objetivo
            action: Acción tomada
            
        Returns:
            Recompensa del paso
        """
        reward = 0.0
        
        # Recompensa por alcanzar el objetivo (MUY GRANDE)
        if self.goal_reached:
            reward += 500.0
            return reward
        
        # Penalización por colisión (GRANDE)
        if self.collision:
            reward -= 200.0
            return reward
        
        # Recompensa por acercarse al objetivo (escalada)
        distance_improvement = prev_dist - curr_dist
        reward += distance_improvement * 50.0  # Más peso al progreso
        
        # Bonus por estar cerca del objetivo
        if curr_dist < 1.0:
            reward += 5.0
        elif curr_dist < 2.0:
            reward += 2.0
        elif curr_dist < 4.0:
            reward += 0.5
        
        # Pequeña penalización por paso (incentiva eficiencia)
        reward -= 0.5
        
        # Penalización por estar muy cerca de obstáculos
        min_obs_dist = self._get_min_obstacle_distance()
        if min_obs_dist < 0.2:
            reward -= 5.0
        elif min_obs_dist < 0.4:
            reward -= 1.0
        
        return reward
    
    def _get_min_obstacle_distance(self) -> float:
        """Obtiene la distancia mínima a cualquier obstáculo"""
        min_dist = float('inf')
        for obs in self.obstacles:
            dist = math.sqrt((self.robot.x - obs.x)**2 + 
                           (self.robot.y - obs.y)**2) - obs.radius
            min_dist = min(min_dist, dist)
        return min_dist
    
    def _get_sensor_distances(self) -> Tuple[float, float, float]:
        """
        Simula sensores de distancia (frente, izquierda, derecha)
        Usa ray casting para detectar obstáculos y paredes
        
        Returns:
            Tupla (distancia_frente, distancia_izquierda, distancia_derecha)
        """
        angles = [
            self.robot.theta,                    # Frente
            self.robot.theta + math.pi/2,        # Izquierda
            self.robot.theta - math.pi/2         # Derecha
        ]
        
        distances = []
        for angle in angles:
            dist = self._cast_ray(angle)
            distances.append(dist)
        
        return tuple(distances)
    
    def _cast_ray(self, angle: float) -> float:
        """
        Lanza un rayo desde el robot y detecta la distancia al primer obstáculo
        
        Args:
            angle: Ángulo del rayo en radianes
            
        Returns:
            Distancia al obstáculo más cercano
        """
        # Dirección del rayo
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        min_dist = self.SENSOR_MAX_RANGE
        
        # Verificar intersección con paredes
        # Pared izquierda (x = 0)
        if dx < 0:
            t = -self.robot.x / dx
            if 0 < t < min_dist:
                y_intersect = self.robot.y + t * dy
                if 0 <= y_intersect <= self.ENV_HEIGHT:
                    min_dist = t
        
        # Pared derecha (x = ENV_WIDTH)
        if dx > 0:
            t = (self.ENV_WIDTH - self.robot.x) / dx
            if 0 < t < min_dist:
                y_intersect = self.robot.y + t * dy
                if 0 <= y_intersect <= self.ENV_HEIGHT:
                    min_dist = t
        
        # Pared inferior (y = 0)
        if dy < 0:
            t = -self.robot.y / dy
            if 0 < t < min_dist:
                x_intersect = self.robot.x + t * dx
                if 0 <= x_intersect <= self.ENV_WIDTH:
                    min_dist = t
        
        # Pared superior (y = ENV_HEIGHT)
        if dy > 0:
            t = (self.ENV_HEIGHT - self.robot.y) / dy
            if 0 < t < min_dist:
                x_intersect = self.robot.x + t * dx
                if 0 <= x_intersect <= self.ENV_WIDTH:
                    min_dist = t
        
        # Verificar intersección con obstáculos circulares
        for obs in self.obstacles:
            # Vector del robot al centro del obstáculo
            ox = obs.x - self.robot.x
            oy = obs.y - self.robot.y
            
            # Proyección sobre el rayo
            proj = ox * dx + oy * dy
            
            if proj < 0:  # Obstáculo detrás del robot
                continue
            
            # Distancia perpendicular al rayo
            perp_dist_sq = ox*ox + oy*oy - proj*proj
            
            if perp_dist_sq < obs.radius * obs.radius:
                # El rayo intersecta el obstáculo
                half_chord = math.sqrt(obs.radius * obs.radius - perp_dist_sq)
                t = proj - half_chord
                if 0 < t < min_dist:
                    min_dist = t
        
        return min_dist
    
    def get_state(self) -> np.ndarray:
        """
        Obtiene el estado actual del robot para enviar al agente
        
        Returns:
            Array con el estado: [x, y, theta, v, omega, d_front, d_left, d_right, dx_goal, dy_goal]
        """
        d_front, d_left, d_right = self._get_sensor_distances()
        
        # Vector relativo al objetivo
        dx_goal = self.goal[0] - self.robot.x
        dy_goal = self.goal[1] - self.robot.y
        
        state = np.array([
            self.robot.x,
            self.robot.y,
            self.robot.theta,
            self.robot.v,
            self.robot.omega,
            d_front,
            d_left,
            d_right,
            dx_goal,
            dy_goal
        ], dtype=np.float32)
        
        return state
    
    def get_state_string(self) -> str:
        """
        Obtiene el estado como string CSV para enviar por TCP
        
        Returns:
            String con formato CSV del estado
        """
        state = self.get_state()
        return ",".join(f"{x:.4f}" for x in state)
    
    def render_ascii(self) -> str:
        """
        Renderiza el entorno en ASCII para visualización simple
        
        Returns:
            String con representación ASCII del entorno
        """
        width = 40
        height = 20
        grid = [['.' for _ in range(width)] for _ in range(height)]
        
        # Dibujar bordes
        for i in range(width):
            grid[0][i] = '#'
            grid[height-1][i] = '#'
        for i in range(height):
            grid[i][0] = '#'
            grid[i][width-1] = '#'
        
        # Dibujar obstáculos
        for obs in self.obstacles:
            gx = int(obs.x / self.ENV_WIDTH * (width - 2)) + 1
            gy = int((1 - obs.y / self.ENV_HEIGHT) * (height - 2)) + 1
            if 1 <= gx < width-1 and 1 <= gy < height-1:
                grid[gy][gx] = 'O'
        
        # Dibujar objetivo
        gx = int(self.goal[0] / self.ENV_WIDTH * (width - 2)) + 1
        gy = int((1 - self.goal[1] / self.ENV_HEIGHT) * (height - 2)) + 1
        if 1 <= gx < width-1 and 1 <= gy < height-1:
            grid[gy][gx] = 'G'
        
        # Dibujar robot
        rx = int(self.robot.x / self.ENV_WIDTH * (width - 2)) + 1
        ry = int((1 - self.robot.y / self.ENV_HEIGHT) * (height - 2)) + 1
        if 1 <= rx < width-1 and 1 <= ry < height-1:
            # Mostrar dirección del robot
            if -math.pi/4 <= self.robot.theta < math.pi/4:
                grid[ry][rx] = '>'
            elif math.pi/4 <= self.robot.theta < 3*math.pi/4:
                grid[ry][rx] = '^'
            elif -3*math.pi/4 <= self.robot.theta < -math.pi/4:
                grid[ry][rx] = 'v'
            else:
                grid[ry][rx] = '<'
        
        return '\n'.join(''.join(row) for row in grid)


# Test básico
if __name__ == "__main__":
    sim = DifferentialRobotSimulator()
    state = sim.reset()
    
    print("=== Test del Simulador de Robot Diferencial ===\n")
    print(f"Estado inicial: {state}")
    print(f"Estado como CSV: {sim.get_state_string()}")
    print(f"\n{sim.render_ascii()}")
    
    print("\n--- Ejecutando 10 pasos con acción 'avanzar' ---\n")
    for i in range(10):
        state, reward, done, info = sim.step(0)  # Avanzar
        print(f"Paso {i+1}: reward={reward:.3f}, dist_goal={info['distance_to_goal']:.2f}")
        if done:
            print(f"Episodio terminado: {info}")
            break
    
    print(f"\n{sim.render_ascii()}")
