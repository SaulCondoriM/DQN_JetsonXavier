"""
Visualizador gráfico con Pygame para el simulador de robot diferencial
"""

import pygame
import math
import sys
from typing import Optional, Tuple
from robot_simulator import DifferentialRobotSimulator, Obstacle


class SimulatorVisualizer:
    """Visualizador 2D usando Pygame"""
    
    # Colores
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 100, 255)
    YELLOW = (255, 255, 0)
    ORANGE = (255, 165, 0)
    DARK_GREEN = (0, 128, 0)
    LIGHT_BLUE = (173, 216, 230)
    
    def __init__(self, simulator: DifferentialRobotSimulator, 
                 window_size: Tuple[int, int] = (800, 800)):
        """
        Inicializa el visualizador
        
        Args:
            simulator: Instancia del simulador
            window_size: Tamaño de la ventana (ancho, alto)
        """
        self.sim = simulator
        self.window_width, self.window_height = window_size
        self.margin = 50  # Margen alrededor del área de simulación
        
        # Calcular escala
        self.scale_x = (self.window_width - 2 * self.margin) / self.sim.ENV_WIDTH
        self.scale_y = (self.window_height - 2 * self.margin) / self.sim.ENV_HEIGHT
        self.scale = min(self.scale_x, self.scale_y)
        
        # Inicializar Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Robot Diferencial - Simulador DQN")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16)
        self.font_large = pygame.font.SysFont('Arial', 24)
        
        # Historial de posiciones para trazar trayectoria
        self.trajectory = []
        self.max_trajectory_points = 500
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convierte coordenadas del mundo a coordenadas de pantalla"""
        screen_x = int(self.margin + x * self.scale)
        screen_y = int(self.window_height - self.margin - y * self.scale)
        return (screen_x, screen_y)
    
    def draw_grid(self):
        """Dibuja una cuadrícula de fondo"""
        for i in range(int(self.sim.ENV_WIDTH) + 1):
            start = self.world_to_screen(i, 0)
            end = self.world_to_screen(i, self.sim.ENV_HEIGHT)
            pygame.draw.line(self.screen, self.GRAY, start, end, 1)
        
        for i in range(int(self.sim.ENV_HEIGHT) + 1):
            start = self.world_to_screen(0, i)
            end = self.world_to_screen(self.sim.ENV_WIDTH, i)
            pygame.draw.line(self.screen, self.GRAY, start, end, 1)
    
    def draw_obstacles(self):
        """Dibuja los obstáculos"""
        for obs in self.sim.obstacles:
            pos = self.world_to_screen(obs.x, obs.y)
            radius = int(obs.radius * self.scale)
            pygame.draw.circle(self.screen, self.RED, pos, radius)
            pygame.draw.circle(self.screen, self.BLACK, pos, radius, 2)
    
    def draw_goal(self):
        """Dibuja el objetivo"""
        pos = self.world_to_screen(self.sim.goal[0], self.sim.goal[1])
        # Círculo exterior
        pygame.draw.circle(self.screen, self.GREEN, pos, int(0.3 * self.scale))
        pygame.draw.circle(self.screen, self.DARK_GREEN, pos, int(0.3 * self.scale), 3)
        # Círculo interior
        pygame.draw.circle(self.screen, self.YELLOW, pos, int(0.15 * self.scale))
        # Texto "GOAL"
        text = self.font.render("GOAL", True, self.BLACK)
        text_rect = text.get_rect(center=(pos[0], pos[1] - 30))
        self.screen.blit(text, text_rect)
    
    def draw_robot(self):
        """Dibuja el robot con su orientación"""
        pos = self.world_to_screen(self.sim.robot.x, self.sim.robot.y)
        radius = int(self.sim.ROBOT_RADIUS * self.scale)
        
        # Cuerpo del robot
        pygame.draw.circle(self.screen, self.BLUE, pos, radius)
        pygame.draw.circle(self.screen, self.BLACK, pos, radius, 2)
        
        # Indicador de dirección (triángulo)
        angle = -self.sim.robot.theta  # Negativo porque Y está invertido en pantalla
        length = radius * 1.2
        end_x = pos[0] + int(length * math.cos(angle))
        end_y = pos[1] + int(length * math.sin(angle))
        
        # Triángulo apuntando hacia adelante
        tip = (end_x, end_y)
        left = (pos[0] + int(radius * 0.5 * math.cos(angle + 2.5)),
                pos[1] + int(radius * 0.5 * math.sin(angle + 2.5)))
        right = (pos[0] + int(radius * 0.5 * math.cos(angle - 2.5)),
                 pos[1] + int(radius * 0.5 * math.sin(angle - 2.5)))
        pygame.draw.polygon(self.screen, self.YELLOW, [tip, left, right])
    
    def draw_sensors(self):
        """Dibuja los rayos de los sensores"""
        robot_pos = self.world_to_screen(self.sim.robot.x, self.sim.robot.y)
        d_front, d_left, d_right = self.sim._get_sensor_distances()
        
        sensors = [
            (d_front, self.sim.robot.theta, self.GREEN),
            (d_left, self.sim.robot.theta + math.pi/2, self.ORANGE),
            (d_right, self.sim.robot.theta - math.pi/2, self.ORANGE)
        ]
        
        for dist, angle, color in sensors:
            end_x = self.sim.robot.x + dist * math.cos(angle)
            end_y = self.sim.robot.y + dist * math.sin(angle)
            end_pos = self.world_to_screen(end_x, end_y)
            pygame.draw.line(self.screen, color, robot_pos, end_pos, 2)
            pygame.draw.circle(self.screen, color, end_pos, 4)
    
    def draw_trajectory(self):
        """Dibuja la trayectoria del robot"""
        if len(self.trajectory) > 1:
            points = [self.world_to_screen(x, y) for x, y in self.trajectory]
            pygame.draw.lines(self.screen, self.LIGHT_BLUE, False, points, 2)
    
    def draw_info(self, action: Optional[int] = None, reward: float = 0.0,
                  episode: int = 0, total_reward: float = 0.0):
        """Dibuja información del estado actual"""
        state = self.sim.get_state()
        
        info_texts = [
            f"Episodio: {episode}",
            f"Paso: {self.sim.episode_steps}",
            f"Posición: ({state[0]:.2f}, {state[1]:.2f})",
            f"Orientación: {math.degrees(state[2]):.1f}°",
            f"Velocidad: v={state[3]:.2f} m/s, ω={state[4]:.2f} rad/s",
            f"Sensores: F={state[5]:.2f}, L={state[6]:.2f}, R={state[7]:.2f}",
            f"Dist. al objetivo: {math.sqrt(state[8]**2 + state[9]**2):.2f} m",
            f"Acción: {self.sim.ACTIONS.get(action, 'N/A') if action is not None else 'N/A'}",
            f"Recompensa: {reward:.3f}",
            f"Recompensa total: {total_reward:.2f}"
        ]
        
        y_offset = 10
        for text in info_texts:
            surface = self.font.render(text, True, self.BLACK)
            self.screen.blit(surface, (10, y_offset))
            y_offset += 20
    
    def update_trajectory(self):
        """Actualiza el historial de trayectoria"""
        self.trajectory.append((self.sim.robot.x, self.sim.robot.y))
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory.pop(0)
    
    def clear_trajectory(self):
        """Limpia la trayectoria"""
        self.trajectory = []
    
    def render(self, action: Optional[int] = None, reward: float = 0.0,
               episode: int = 0, total_reward: float = 0.0):
        """
        Renderiza un frame del simulador
        
        Args:
            action: Última acción tomada
            reward: Última recompensa obtenida
            episode: Número de episodio actual
            total_reward: Recompensa acumulada del episodio
        """
        # Manejar eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Limpiar pantalla
        self.screen.fill(self.WHITE)
        
        # Dibujar elementos
        self.draw_grid()
        self.draw_trajectory()
        self.draw_obstacles()
        self.draw_goal()
        self.draw_sensors()
        self.draw_robot()
        self.draw_info(action, reward, episode, total_reward)
        
        # Actualizar pantalla
        pygame.display.flip()
        
        # Controlar FPS
        self.clock.tick(30)
    
    def close(self):
        """Cierra el visualizador"""
        pygame.quit()


# Demo interactivo
if __name__ == "__main__":
    sim = DifferentialRobotSimulator()
    vis = SimulatorVisualizer(sim)
    
    print("=== Demo Interactivo del Simulador ===")
    print("Controles:")
    print("  W - Avanzar")
    print("  A - Girar izquierda")
    print("  D - Girar derecha")
    print("  S - Frenar")
    print("  X - Retroceder")
    print("  R - Reiniciar episodio")
    print("  Q - Salir")
    
    sim.reset()
    vis.clear_trajectory()
    
    episode = 0
    total_reward = 0.0
    running = True
    action = None
    reward = 0.0
    
    while running:
        # Procesar entrada
        keys = pygame.key.get_pressed()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    episode += 1
                    sim.reset()
                    vis.clear_trajectory()
                    total_reward = 0.0
                    print(f"\n--- Episodio {episode} ---")
        
        # Determinar acción basada en teclas presionadas
        action = None
        if keys[pygame.K_w]:
            action = 0  # Avanzar
        elif keys[pygame.K_a]:
            action = 1  # Girar izquierda
        elif keys[pygame.K_d]:
            action = 2  # Girar derecha
        elif keys[pygame.K_s]:
            action = 3  # Frenar
        elif keys[pygame.K_x]:
            action = 4  # Retroceder
        
        if action is not None:
            state, reward, done, info = sim.step(action)
            total_reward += reward
            vis.update_trajectory()
            
            if done:
                if info['goal_reached']:
                    print(f"¡OBJETIVO ALCANZADO! Recompensa total: {total_reward:.2f}")
                elif info['collision']:
                    print(f"¡COLISIÓN! Recompensa total: {total_reward:.2f}")
                else:
                    print(f"Episodio terminado. Recompensa total: {total_reward:.2f}")
        
        vis.render(action, reward, episode, total_reward)
    
    vis.close()
