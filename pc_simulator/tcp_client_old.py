"""
Cliente TCP para comunicación con el agente DQN en Jetson Xavier
El PC actúa como cliente, enviando estados y recibiendo acciones
"""

import socket
import time
import struct
from typing import Optional, Tuple
from robot_simulator import DifferentialRobotSimulator


class TCPClient:
    """Cliente TCP para comunicación con el agente en Jetson"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, 
                 timeout: float = 0.5):
        """
        Inicializa el cliente TCP
        
        Args:
            host: IP del servidor (Jetson Xavier)
            port: Puerto del servidor
            timeout: Timeout para operaciones de socket
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.buffer_size = 4096
    
    def connect(self, max_retries: int = 5, retry_delay: float = 2.0) -> bool:
        """
        Conecta al servidor en el Jetson
        
        Args:
            max_retries: Número máximo de intentos de conexión
            retry_delay: Tiempo entre reintentos en segundos
            
        Returns:
            True si la conexión fue exitosa
        """
        for attempt in range(max_retries):
            try:
                print(f"Intentando conectar a {self.host}:{self.port} "
                      f"(intento {attempt + 1}/{max_retries})...")
                
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(self.timeout)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.socket.connect((self.host, self.port))
                
                self.connected = True
                print(f"✓ Conectado exitosamente a {self.host}:{self.port}")
                return True
                
            except socket.timeout:
                print(f"  Timeout al conectar")
            except socket.error as e:
                print(f"  Error de conexión: {e}")
            except Exception as e:
                print(f"  Error inesperado: {e}")
            
            if attempt < max_retries - 1:
                print(f"  Reintentando en {retry_delay} segundos...")
                time.sleep(retry_delay)
        
        print(f"✗ No se pudo conectar después de {max_retries} intentos")
        return False
    
    def disconnect(self):
        """Cierra la conexión"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None
        self.connected = False
        print("Desconectado del servidor")
    
    def send_state(self, state_str: str) -> bool:
        """
        Envía el estado del robot al Jetson
        
        Args:
            state_str: Estado en formato CSV
            
        Returns:
            True si el envío fue exitoso
        """
        if not self.connected or not self.socket:
            return False
        
        try:
            # Agregar newline como delimitador
            message = state_str + "\n"
            self.socket.sendall(message.encode('utf-8'))
            return True
        except socket.error as e:
            print(f"Error al enviar estado: {e}")
            self.connected = False
            return False
    
    def receive_action(self) -> Optional[int]:
        """
        Recibe la acción del Jetson
        
        Returns:
            Índice de acción (0-4) o None si hubo error
        """
        if not self.connected or not self.socket:
            return None
        
        try:
            data = b""
            while b"\n" not in data:
                try:
                    chunk = self.socket.recv(self.buffer_size)
                except socket.timeout:
                    # Caller will poll again; return None to indicate no data yet
                    return None
                if not chunk:
                    print("Conexión cerrada por el servidor")
                    self.connected = False
                    return None
                data += chunk

            # Parsear la acción
            action_str = data.decode('utf-8').strip()
            action = int(action_str)

            if 0 <= action <= 4:
                return action
            else:
                print(f"Acción inválida recibida: {action}")
                return None

        except ValueError as e:
            print(f"Error parseando acción: {e}")
            return None
        except socket.error as e:
            print(f"Error de socket: {e}")
            self.connected = False
            return None
    
    def send_episode_info(self, done: bool, reward: float, 
                          goal_reached: bool, collision: bool) -> bool:
        """
        Envía información adicional del episodio
        
        Args:
            done: Si el episodio terminó
            reward: Recompensa del paso
            goal_reached: Si se alcanzó el objetivo
            collision: Si hubo colisión
            
        Returns:
            True si el envío fue exitoso
        """
        if not self.connected or not self.socket:
            return False
        
        try:
            # Formato: done,reward,goal_reached,collision
            info_str = f"{int(done)},{reward:.6f},{int(goal_reached)},{int(collision)}\n"
            self.socket.sendall(info_str.encode('utf-8'))
            return True
        except socket.error as e:
            print(f"Error al enviar info: {e}")
            return False
    
    def step_communication(self, state_str: str, done: bool, reward: float,
                          goal_reached: bool, collision: bool) -> Optional[int]:
        """
        Realiza un ciclo completo de comunicación:
        1. Envía estado
        2. Envía info del episodio
        3. Recibe acción
        
        Args:
            state_str: Estado en formato CSV
            done: Si el episodio terminó
            reward: Recompensa del paso anterior
            goal_reached: Si se alcanzó el objetivo
            collision: Si hubo colisión
            
        Returns:
            Acción recibida o None si hubo error
        """
        # Enviar estado + info en un solo mensaje
        # Formato: estado|done,reward,goal,collision
        combined = f"{state_str}|{int(done)},{reward:.6f},{int(goal_reached)},{int(collision)}\n"
        
        try:
            self.socket.sendall(combined.encode('utf-8'))
            return self.receive_action()
        except socket.error as e:
            print(f"Error en comunicación: {e}")
            self.connected = False
            return None


class SimulatorTCPClient:
    """
    Cliente que integra el simulador con la comunicación TCP
    """
    
    def __init__(self, jetson_ip: str = "127.0.0.1", port: int = 5555):
        """
        Args:
            jetson_ip: IP del Jetson Xavier
            port: Puerto de comunicación
        """
        self.simulator = DifferentialRobotSimulator()
        self.tcp_client = TCPClient(host=jetson_ip, port=port)
        self.visualizer = None
        self.use_visualization = False
    
    def enable_visualization(self):
        """Habilita la visualización con Pygame"""
        try:
            from visualizer import SimulatorVisualizer
            self.visualizer = SimulatorVisualizer(self.simulator)
            self.use_visualization = True
            print("Visualización habilitada")
        except ImportError as e:
            print(f"No se pudo habilitar visualización: {e}")
            self.use_visualization = False
    
    def run_training(self, num_episodes: int = 1000, 
                     render_every: int = 10,
                     print_every: int = 1):
        """
        Ejecuta el loop principal de entrenamiento
        
        Args:
            num_episodes: Número de episodios a ejecutar
            render_every: Renderizar cada N episodios
            print_every: Imprimir estadísticas cada N episodios
        """
        if not self.tcp_client.connect():
            print("No se pudo conectar al Jetson. Abortando.")
            return
        
        try:
            episode_rewards = []
            episode_lengths = []
            goals_reached = 0
            collisions = 0
            
            for episode in range(num_episodes):
                # Reiniciar episodio
                state = self.simulator.reset()
                if self.use_visualization:
                    self.visualizer.clear_trajectory()
                
                total_reward = 0.0
                done = False
                step = 0
                prev_reward = 0.0
                
                # Enviar estado inicial
                state_str = self.simulator.get_state_string()
                # Enviar estado inicial y esperar acción (polling)
                state_str = self.simulator.get_state_string()
                # step_communication envía el mensaje y trata de leer una acción;
                # puede devolver None si aún no hay respuesta (timeout). Hacemos polling
                action = self.tcp_client.step_communication(
                    state_str, False, 0.0, False, False
                )
                # Poll hasta recibir acción o hasta timeout
                poll_start = time.time()
                poll_timeout = 2.0  # segundos
                while action is None and time.time() - poll_start < poll_timeout:
                    # Mantener GUI responsiva
                    if self.use_visualization:
                        try:
                            import pygame
                            pygame.event.pump()
                        except Exception:
                            pass
                    time.sleep(0.01)
                    action = self.tcp_client.receive_action()

                if action is None:
                    print("Error en comunicación inicial (timeout) - usando acción 'frenar'")
                    action = 3
                
                while not done:
                    # Ejecutar acción en el simulador
                    state, reward, done, info = self.simulator.step(action)
                    total_reward += reward
                    step += 1
                    
                    # Visualizar si corresponde
                    if self.use_visualization and episode % render_every == 0:
                        self.visualizer.update_trajectory()
                        self.visualizer.render(action, reward, episode, total_reward)
                    
                    # Obtener nueva acción del Jetson (envío y polling)
                    state_str = self.simulator.get_state_string()
                    action = self.tcp_client.step_communication(
                        state_str, done, reward,
                        info['goal_reached'], info['collision']
                    )
                    poll_start = time.time()
                    poll_timeout = 2.0
                    while action is None and time.time() - poll_start < poll_timeout:
                        if self.use_visualization:
                            try:
                                import pygame
                                pygame.event.pump()
                            except Exception:
                                pass
                        time.sleep(0.01)
                        action = self.tcp_client.receive_action()

                    if action is None:
                        # No se recibió acción en el tiempo esperado; evitar abortar el episodio
                        print("Warning: no se recibió acción del Jetson (timeout). Aplicando 'frenar'.")
                        action = 3
                    
                    prev_reward = reward
                
                # Estadísticas del episodio
                episode_rewards.append(total_reward)
                episode_lengths.append(step)
                
                if info.get('goal_reached', False):
                    goals_reached += 1
                if info.get('collision', False):
                    collisions += 1
                
                # Imprimir progreso
                if (episode + 1) % print_every == 0:
                    avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
                    avg_length = sum(episode_lengths[-100:]) / min(len(episode_lengths), 100)
                    success_rate = goals_reached / (episode + 1) * 100
                    
                    print(f"Episodio {episode + 1}/{num_episodes} | "
                          f"Recompensa: {total_reward:.2f} | "
                          f"Promedio(100): {avg_reward:.2f} | "
                          f"Pasos: {step} | "
                          f"Éxitos: {success_rate:.1f}%")
            
            # Resumen final
            print("\n=== Resumen del Entrenamiento ===")
            print(f"Episodios completados: {len(episode_rewards)}")
            print(f"Objetivos alcanzados: {goals_reached} ({goals_reached/len(episode_rewards)*100:.1f}%)")
            print(f"Colisiones: {collisions} ({collisions/len(episode_rewards)*100:.1f}%)")
            print(f"Recompensa promedio: {sum(episode_rewards)/len(episode_rewards):.2f}")
            print(f"Longitud promedio: {sum(episode_lengths)/len(episode_lengths):.1f} pasos")
            
        except KeyboardInterrupt:
            print("\nEntrenamiento interrumpido por el usuario")
        finally:
            self.tcp_client.disconnect()
            if self.visualizer:
                self.visualizer.close()


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulador de Robot - Cliente TCP')
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                        help='IP del Jetson Xavier (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5555,
                        help='Puerto TCP (default: 5555)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Número de episodios (default: 1000)')
    parser.add_argument('--visualize', action='store_true',
                        help='Habilitar visualización')
    parser.add_argument('--render-every', type=int, default=10,
                        help='Renderizar cada N episodios (default: 10)')
    
    args = parser.parse_args()
    
    print("=== Simulador de Robot Diferencial - Cliente TCP ===")
    print(f"Conectando a Jetson en {args.ip}:{args.port}")
    
    client = SimulatorTCPClient(jetson_ip=args.ip, port=args.port)
    
    if args.visualize:
        client.enable_visualization()
    
    client.run_training(
        num_episodes=args.episodes,
        render_every=args.render_every
    )


if __name__ == "__main__":
    main()
