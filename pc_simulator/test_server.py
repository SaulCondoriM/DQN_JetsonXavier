"""
Servidor de prueba en Python que simula al Jetson
Usa un agente DQN simple para verificar la comunicación
"""

import socket
import numpy as np
import random
import threading
import time
from collections import deque


class SimpleReplayBuffer:
    """Replay buffer simple para pruebas"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class SimpleDQN:
    """
    DQN simple usando NumPy para pruebas locales
    Simula el comportamiento del agente en el Jetson
    """
    def __init__(self, state_size=10, action_size=5, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size
        
        # Pesos de la red (simple MLP)
        self.W1 = np.random.randn(state_size, hidden_size) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.W3 = np.random.randn(hidden_size, action_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros(action_size)
        
        # Target network (copia)
        self.W1_target = self.W1.copy()
        self.b1_target = self.b1.copy()
        self.W2_target = self.W2.copy()
        self.b2_target = self.b2.copy()
        self.W3_target = self.W3.copy()
        self.b3_target = self.b3.copy()
        
        # Hiperparámetros
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.batch_size = 64
        self.tau = 0.005
        
        # Replay buffer
        self.replay_buffer = SimpleReplayBuffer(100000)
        self.min_buffer_size = 1000
        
        # Estadísticas
        self.train_steps = 0
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, state, use_target=False):
        """Forward pass"""
        if use_target:
            h1 = self.relu(np.dot(state, self.W1_target) + self.b1_target)
            h2 = self.relu(np.dot(h1, self.W2_target) + self.b2_target)
            q_values = np.dot(h2, self.W3_target) + self.b3_target
        else:
            h1 = self.relu(np.dot(state, self.W1) + self.b1)
            h2 = self.relu(np.dot(h1, self.W2) + self.b2)
            q_values = np.dot(h2, self.W3) + self.b3
        return q_values
    
    def select_action(self, state):
        """Selección epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state = np.array(state).reshape(1, -1)
        q_values = self.forward(state)
        return np.argmax(q_values)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Almacena transición"""
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train_step(self):
        """Un paso de entrenamiento"""
        if len(self.replay_buffer) < self.min_buffer_size:
            return 0.0
        
        # Muestrear batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # Calcular targets
        next_q_values = self.forward(next_states, use_target=True)
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Forward pass
        h1 = self.relu(np.dot(states, self.W1) + self.b1)
        h2 = self.relu(np.dot(h1, self.W2) + self.b2)
        q_values = np.dot(h2, self.W3) + self.b3
        
        # Calcular error solo para las acciones tomadas
        batch_indices = np.arange(self.batch_size)
        q_predicted = q_values[batch_indices, actions]
        
        # Loss y gradiente
        errors = q_predicted - targets
        loss = np.mean(errors ** 2)
        
        # Backprop simplificado (gradient descent básico)
        d_output = np.zeros_like(q_values)
        d_output[batch_indices, actions] = 2 * errors / self.batch_size
        
        # Gradientes
        d_W3 = np.dot(h2.T, d_output)
        d_b3 = np.sum(d_output, axis=0)
        
        d_h2 = np.dot(d_output, self.W3.T) * (h2 > 0)
        d_W2 = np.dot(h1.T, d_h2)
        d_b2 = np.sum(d_h2, axis=0)
        
        d_h1 = np.dot(d_h2, self.W2.T) * (h1 > 0)
        d_W1 = np.dot(states.T, d_h1)
        d_b1 = np.sum(d_h1, axis=0)
        
        # Actualizar pesos
        self.W3 -= self.learning_rate * d_W3
        self.b3 -= self.learning_rate * d_b3
        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2
        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1
        
        # Soft update de target network
        self.W1_target = self.tau * self.W1 + (1 - self.tau) * self.W1_target
        self.b1_target = self.tau * self.b1 + (1 - self.tau) * self.b1_target
        self.W2_target = self.tau * self.W2 + (1 - self.tau) * self.W2_target
        self.b2_target = self.tau * self.b2 + (1 - self.tau) * self.b2_target
        self.W3_target = self.tau * self.W3 + (1 - self.tau) * self.W3_target
        self.b3_target = self.tau * self.b3 + (1 - self.tau) * self.b3_target
        
        self.train_steps += 1
        return loss


class TestServer:
    """Servidor TCP de prueba que simula al Jetson"""
    
    def __init__(self, port=5555):
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.agent = SimpleDQN()
        self.running = False
    
    def start(self):
        """Inicia el servidor"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(1)
        
        print(f"[TestServer] Escuchando en puerto {self.port}...")
        print(f"[TestServer] Esperando conexión del simulador...")
        
        self.client_socket, addr = self.server_socket.accept()
        self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[TestServer] Cliente conectado desde: {addr}")
        
        self.running = True
        return True
    
    def receive_state(self):
        """Recibe estado y info del simulador"""
        try:
            data = b""
            while b"\n" not in data:
                chunk = self.client_socket.recv(4096)
                if not chunk:
                    return None, None
                data += chunk
            
            message = data.decode('utf-8').strip()
            
            # Parsear: estado|done,reward,goal,collision
            parts = message.split('|')
            if len(parts) != 2:
                return None, None
            
            state_str, info_str = parts
            
            # Parsear estado
            state = [float(x) for x in state_str.split(',')]
            
            # Parsear info
            info_parts = info_str.split(',')
            info = {
                'done': int(info_parts[0]) != 0,
                'reward': float(info_parts[1]),
                'goal_reached': int(info_parts[2]) != 0,
                'collision': int(info_parts[3]) != 0
            }
            
            return state, info
            
        except Exception as e:
            print(f"[TestServer] Error recibiendo estado: {e}")
            return None, None
    
    def send_action(self, action):
        """Envía acción al simulador"""
        try:
            message = f"{action}\n"
            self.client_socket.sendall(message.encode('utf-8'))
            return True
        except Exception as e:
            print(f"[TestServer] Error enviando acción: {e}")
            return False
    
    def run(self, max_episodes=100):
        """Loop principal del servidor"""
        if not self.start():
            return
        
        episode = 0
        total_goals = 0
        total_collisions = 0
        
        print("\n" + "="*50)
        print("   DQN Test Server (Python)")
        print("="*50 + "\n")
        
        try:
            while self.running and episode < max_episodes:
                # Recibir estado inicial
                state, info = self.receive_state()
                if state is None:
                    break
                
                episode += 1
                episode_steps = 0
                episode_reward = 0.0
                prev_state = None
                prev_action = None
                
                # Primera acción
                action = self.agent.select_action(state)
                self.send_action(action)
                
                while self.running:
                    prev_state = state
                    prev_action = action
                    
                    # Recibir nuevo estado
                    state, info = self.receive_state()
                    if state is None:
                        self.running = False
                        break
                    
                    episode_steps += 1
                    episode_reward += info['reward']
                    
                    # Almacenar transición
                    self.agent.store_transition(
                        prev_state, prev_action, info['reward'],
                        state, info['done']
                    )
                    
                    # Entrenar
                    if episode_steps % 4 == 0:
                        self.agent.train_step()
                    
                    # Verificar si terminó
                    if info['done']:
                        if info['goal_reached']:
                            total_goals += 1
                            result = "✓ GOAL"
                        elif info['collision']:
                            total_collisions += 1
                            result = "✗ COLLISION"
                        else:
                            result = "- TIMEOUT"
                        
                        print(f"Ep {episode:4d} | Steps: {episode_steps:4d} | "
                              f"Reward: {episode_reward:8.2f} | {result} | "
                              f"ε: {self.agent.epsilon:.4f}")
                        break
                    
                    # Seleccionar siguiente acción
                    action = self.agent.select_action(state)
                    self.send_action(action)
                
                # Enviar última acción (aunque el episodio terminó)
                action = self.agent.select_action(state)
                self.send_action(action)
        
        except KeyboardInterrupt:
            print("\n[TestServer] Interrumpido por usuario")
        
        finally:
            print(f"\n=== Resumen ===")
            print(f"Episodios: {episode}")
            print(f"Objetivos: {total_goals} ({100*total_goals/max(1,episode):.1f}%)")
            print(f"Colisiones: {total_collisions} ({100*total_collisions/max(1,episode):.1f}%)")
            
            if self.client_socket:
                self.client_socket.close()
            if self.server_socket:
                self.server_socket.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Servidor de prueba DQN')
    parser.add_argument('--port', type=int, default=5555, help='Puerto TCP')
    parser.add_argument('--episodes', type=int, default=100, help='Número de episodios')
    args = parser.parse_args()
    
    server = TestServer(port=args.port)
    server.run(max_episodes=args.episodes)


if __name__ == "__main__":
    main()
