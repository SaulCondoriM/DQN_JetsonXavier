#!/usr/bin/env python3
"""
Cliente TCP con comunicacion robusta para el sistema DQN distribuido.
El PC actua como cliente, enviando estados al Jetson y recibiendo acciones.
"""

import socket
import time
import argparse
from typing import Optional, Tuple, List
from robot_simulator import DifferentialRobotSimulator


class TCPClient:
    """Cliente TCP robusto para comunicacion con el agente DQN en Jetson"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5555):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.recv_buffer = ""
        
    def connect(self, max_retries: int = 5, retry_delay: float = 2.0) -> bool:
        """Conecta al servidor en el Jetson"""
        for attempt in range(max_retries):
            try:
                print(f"Intentando conectar a {self.host}:{self.port} "
                      f"(intento {attempt + 1}/{max_retries})...")
                
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.socket.settimeout(10.0)
                self.socket.connect((self.host, self.port))
                
                self.connected = True
                print(f"Conectado exitosamente a {self.host}:{self.port}")
                return True
                
            except socket.timeout:
                print(f"Timeout al conectar")
            except socket.error as e:
                print(f"Error de socket: {e}")
            
            if attempt < max_retries - 1:
                print(f"Reintentando en {retry_delay}s...")
                time.sleep(retry_delay)
        
        print("No se pudo conectar al servidor")
        return False
    
    def disconnect(self):
        """Cierra la conexion"""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        print("Desconectado del servidor")
    
    def send_state(self, state: List[float], done: bool, reward: float,
                   goal_reached: bool, collision: bool) -> bool:
        """
        Envia el estado y la informacion del episodio al servidor.
        Formato: "s1,s2,s3,...,s10|done,reward,goal,collision\n"
        """
        if not self.connected or not self.socket:
            return False
        
        # Validar que el estado tenga exactamente 10 valores
        if len(state) != 10:
            print(f"[ERROR] Estado invalido: {len(state)} valores (esperados 10)")
            return False
        
        # Construir mensaje de forma atomica
        state_str = ",".join(f"{v:.6f}" for v in state)
        info_str = f"{int(done)},{reward:.4f},{int(goal_reached)},{int(collision)}"
        message = f"{state_str}|{info_str}\n"
        
        try:
            self.socket.sendall(message.encode('utf-8'))
            return True
        except socket.error as e:
            print(f"[ERROR] Error al enviar estado: {e}")
            self.connected = False
            return False
    
    def receive_action(self, timeout: float = 5.0) -> Optional[int]:
        """
        Recibe la accion del servidor.
        Returns: Indice de la accion (0-4) o None si hay error
        """
        if not self.connected or not self.socket:
            return None
        
        try:
            self.socket.settimeout(timeout)
            
            # Leer hasta encontrar newline
            while '\n' not in self.recv_buffer:
                data = self.socket.recv(1024)
                if not data:
                    print("[WARN] Servidor cerro la conexion")
                    self.connected = False
                    return None
                self.recv_buffer += data.decode('utf-8')
            
            # Extraer primera linea completa
            newline_pos = self.recv_buffer.index('\n')
            line = self.recv_buffer[:newline_pos].strip()
            self.recv_buffer = self.recv_buffer[newline_pos + 1:]
            
            action = int(line)
            if 0 <= action <= 4:
                return action
            else:
                print(f"[WARN] Accion fuera de rango: {action}")
                return None
                
        except socket.timeout:
            print("[WARN] Timeout esperando accion")
            return None
        except ValueError as e:
            print(f"[ERROR] Error parseando accion: {e}")
            return None
        except socket.error as e:
            print(f"[ERROR] Error de socket: {e}")
            self.connected = False
            return None


class DQNTrainingClient:
    """Cliente de entrenamiento DQN con visualizacion opcional"""
    
    def __init__(self, host: str, port: int, visualize: bool = True):
        self.client = TCPClient(host, port)
        self.visualize = visualize
        self.simulator = DifferentialRobotSimulator()
        self.visualizer = None
        
        # Estadisticas
        self.episode_rewards = []
        self.goals_reached = 0
        self.collisions = 0
        
    def setup_visualization(self):
        """Inicializa la visualizacion si esta habilitada"""
        if self.visualize:
            try:
                from visualizer import SimulatorVisualizer
                self.visualizer = SimulatorVisualizer(self.simulator)
                print("Visualizacion habilitada")
            except ImportError as e:
                print(f"[WARN] No se pudo cargar visualizador: {e}")
                self.visualize = False
    
    def get_state_vector(self) -> List[float]:
        """Obtiene el vector de estado del simulador"""
        # get_state() devuelve: [x, y, theta, v, omega, d_front, d_left, d_right, dx_goal, dy_goal]
        state = self.simulator.get_state()
        
        # Normalizar los valores
        normalized = [
            state[0] / self.simulator.ENV_WIDTH,      # x
            state[1] / self.simulator.ENV_HEIGHT,     # y
            state[2] / 3.14159,                        # theta
            state[3] / self.simulator.MAX_LINEAR_VEL, # v
            state[4] / self.simulator.MAX_ANGULAR_VEL,# omega
            state[5] / self.simulator.SENSOR_MAX_RANGE, # d_front
            state[6] / self.simulator.SENSOR_MAX_RANGE, # d_left
            state[7] / self.simulator.SENSOR_MAX_RANGE, # d_right
            state[8] / self.simulator.ENV_WIDTH,      # dx_goal
            state[9] / self.simulator.ENV_HEIGHT      # dy_goal
        ]
        return normalized
    
    def run_episode(self, episode_num: int, max_steps: int = 1000) -> Tuple[float, bool, bool, bool]:
        """
        Ejecuta un episodio de entrenamiento
        Returns: (total_reward, goal_reached, collision, success) - success indica si el episodio terminó correctamente
        """
        self.simulator.reset()
        total_reward = 0.0
        step_reward = 0.0
        
        for step in range(max_steps):
            state = self.get_state_vector()
            done = self.simulator.collision or self.simulator.goal_reached
            
            # Enviar estado
            if not self.client.send_state(state, done, step_reward, 
                                          self.simulator.goal_reached, 
                                          self.simulator.collision):
                print(f"[ERROR] Fallo al enviar estado en paso {step}")
                return total_reward, self.simulator.goal_reached, self.simulator.collision, False
            
            # Si terminó, recibir última acción y salir
            if done:
                _ = self.client.receive_action(timeout=5.0)
                return total_reward, self.simulator.goal_reached, self.simulator.collision, True
            
            # Recibir acción
            action = self.client.receive_action(timeout=15.0)
            if action is None:
                print(f"[ERROR] No se recibio accion en paso {step}")
                return total_reward, self.simulator.goal_reached, self.simulator.collision, False
            
            # Ejecutar paso
            _, step_reward, done, info = self.simulator.step(action)
            total_reward += step_reward
            
            # Visualización
            if self.visualizer:
                try:
                    self.visualizer.update_trajectory()
                    self.visualizer.render()
                except SystemExit:
                    return total_reward, self.simulator.goal_reached, self.simulator.collision, False
        
        # Timeout por max_steps - enviar estado final
        state = self.get_state_vector()
        self.client.send_state(state, True, step_reward, False, False)
        _ = self.client.receive_action(timeout=5.0)
        return total_reward, False, False, True
    
    def run_training(self, num_episodes: int = 1000):
        """Ejecuta el entrenamiento completo"""
        print("\n" + "="*60)
        print("INICIANDO ENTRENAMIENTO DQN DISTRIBUIDO")
        print("="*60)
        print(f"Episodios: {num_episodes}")
        print(f"Visualizacion: {'Habilitada' if self.visualize else 'Deshabilitada'}")
        print("="*60 + "\n")
        
        if not self.client.connect():
            print("No se pudo conectar al servidor. Abortando.")
            return
        
        self.setup_visualization()
        
        try:
            for episode in range(num_episodes):
                reward, goal, collision, success = self.run_episode(episode, max_steps=1000)
                
                if not success:
                    print(f"\n[ERROR] Episodio {episode+1} falló. Deteniendo entrenamiento.")
                    print("        (El Jetson y el PC están desincronizados)")
                    break
                
                self.episode_rewards.append(reward)
                if goal:
                    self.goals_reached += 1
                if collision:
                    self.collisions += 1
                
                avg_reward = sum(self.episode_rewards[-100:]) / min(len(self.episode_rewards), 100)
                success_rate = self.goals_reached / (episode + 1) * 100
                
                # Mostrar cada episodio para mejor seguimiento
                result = "GOAL!" if goal else ("COLLISION" if collision else "TIMEOUT")
                print(f"Ep {episode + 1:4d} | "
                      f"Reward: {reward:8.2f} | "
                      f"Avg: {avg_reward:8.2f} | "
                      f"{result} | "
                      f"Goals: {self.goals_reached} ({success_rate:.1f}%)")
                
                if not self.client.connected:
                    print("\n[ERROR] Conexion perdida. Deteniendo entrenamiento.")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nEntrenamiento interrumpido por el usuario")
        finally:
            self.client.disconnect()
            if self.visualizer:
                self.visualizer.close()
        
        print("\n" + "="*60)
        print("RESUMEN DEL ENTRENAMIENTO")
        print("="*60)
        print(f"Episodios completados: {len(self.episode_rewards)}")
        print(f"Objetivos alcanzados: {self.goals_reached}")
        print(f"Colisiones: {self.collisions}")
        if self.episode_rewards:
            print(f"Recompensa promedio: {sum(self.episode_rewards)/len(self.episode_rewards):.2f}")
            print(f"Mejor recompensa: {max(self.episode_rewards):.2f}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Cliente TCP para entrenamiento DQN')
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                        help='IP del servidor Jetson (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5555,
                        help='Puerto del servidor (default: 5555)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Numero de episodios (default: 1000)')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Desactivar visualizacion')
    
    args = parser.parse_args()
    
    client = DQNTrainingClient(
        host=args.ip,
        port=args.port,
        visualize=not args.no_visualize
    )
    
    client.run_training(num_episodes=args.episodes)


if __name__ == "__main__":
    main()
