/**
 * @file dqn_agent.cuh
 * @brief Agente DQN completo con entrenamiento online
 * 
 * Implementación del algoritmo DQN con:
 * - Red principal y target network
 * - Experience replay
 * - Política epsilon-greedy
 * - Entrenamiento online
 */

#ifndef DQN_AGENT_CUH
#define DQN_AGENT_CUH

#include "neural_network.cuh"
#include "replay_buffer.hpp"
#include <random>
#include <chrono>
#include <algorithm>

/**
 * @brief Configuración del agente DQN
 */
struct DQNConfig {
    // Arquitectura
    int state_size = 10;
    int action_size = 5;
    std::vector<int> hidden_sizes = {128, 128};
    
    // Entrenamiento
    float learning_rate = 0.001f;
    float gamma = 0.99f;  // Factor de descuento
    int batch_size = 64;
    int replay_buffer_size = 100000;
    int min_replay_size = 1000;  // Mínimo de experiencias antes de entrenar
    
    // Epsilon-greedy
    float epsilon_start = 1.0f;
    float epsilon_end = 0.01f;
    float epsilon_decay = 0.995f;
    
    // Target network
    float tau = 0.005f;  // Factor de soft update
    int target_update_freq = 100;  // Frecuencia de actualización (pasos)
    bool use_soft_update = true;
    
    // Adam optimizer
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_epsilon = 1e-8f;
    
    // Misc
    int train_freq = 4;  // Entrenar cada N pasos
    bool verbose = true;
};

/**
 * @brief Agente DQN con soporte CUDA
 */
class DQNAgent {
private:
    DQNConfig config;
    
    // Redes neuronales
    DQNNetwork* q_network;
    DQNNetwork* target_network;
    
    // Replay buffer
    ReplayBuffer* replay_buffer;
    
    // Estado del agente
    float epsilon;
    int total_steps;
    int train_steps;
    int episodes;
    
    // Buffers GPU para entrenamiento
    float* d_states;
    float* d_next_states;
    float* d_q_values;
    float* d_next_q_values;
    float* d_target_q;
    float* d_loss_grad;
    
    // RNG
    std::mt19937 rng;
    std::uniform_real_distribution<float> uniform_dist;
    
    // Estadísticas
    float total_loss;
    int loss_count;
    
public:
    DQNAgent(const DQNConfig& cfg) : config(cfg) {
        // Crear redes
        q_network = new DQNNetwork(cfg.state_size, cfg.action_size,
                                    cfg.hidden_sizes, cfg.batch_size);
        target_network = new DQNNetwork(cfg.state_size, cfg.action_size,
                                         cfg.hidden_sizes, cfg.batch_size);
        
        // Copiar pesos iniciales a target network
        q_network->copy_weights_to(*target_network);
        
        // Crear replay buffer
        replay_buffer = new ReplayBuffer(cfg.replay_buffer_size, cfg.state_size);
        
        // Inicializar estado
        epsilon = cfg.epsilon_start;
        total_steps = 0;
        train_steps = 0;
        episodes = 0;
        total_loss = 0.0f;
        loss_count = 0;
        
        // Allocar buffers GPU
        int batch_state_size = cfg.batch_size * cfg.state_size;
        int batch_action_size = cfg.batch_size * cfg.action_size;
        
        CUDA_CHECK(cudaMalloc(&d_states, batch_state_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_next_states, batch_state_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q_values, batch_action_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_next_q_values, batch_action_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_target_q, batch_action_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_loss_grad, batch_action_size * sizeof(float)));
        
        // Inicializar RNG
        std::random_device rd;
        rng.seed(rd());
        uniform_dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
        
        if (config.verbose) {
            printf("[DQNAgent] Inicializado\n");
            printf("  - State size: %d\n", cfg.state_size);
            printf("  - Action size: %d\n", cfg.action_size);
            printf("  - Hidden layers: ");
            for (int h : cfg.hidden_sizes) printf("%d ", h);
            printf("\n");
            printf("  - Learning rate: %.6f\n", cfg.learning_rate);
            printf("  - Gamma: %.3f\n", cfg.gamma);
            printf("  - Epsilon: %.3f -> %.3f\n", cfg.epsilon_start, cfg.epsilon_end);
        }
    }
    
    ~DQNAgent() {
        delete q_network;
        delete target_network;
        delete replay_buffer;
        
        cudaFree(d_states);
        cudaFree(d_next_states);
        cudaFree(d_q_values);
        cudaFree(d_next_q_values);
        cudaFree(d_target_q);
        cudaFree(d_loss_grad);
    }
    
    /**
     * @brief Selecciona una acción usando política epsilon-greedy
     * @param state Estado actual
     * @return Índice de la acción seleccionada
     */
    int select_action(const float* state) {
        // Exploración
        if (uniform_dist(rng) < epsilon) {
            std::uniform_int_distribution<int> action_dist(0, config.action_size - 1);
            return action_dist(rng);
        }
        
        // Explotación: usar Q-network
        return q_network->get_best_action(state);
    }
    
    /**
     * @brief Almacena una transición en el replay buffer
     */
    void store_transition(const float* state, int action, float reward,
                         const float* next_state, bool done) {
        replay_buffer->add(state, action, reward, next_state, done);
        total_steps++;
        
        // Actualizar epsilon
        if (epsilon > config.epsilon_end) {
            epsilon *= config.epsilon_decay;
            epsilon = std::max(epsilon, config.epsilon_end);
        }
    }
    
    /**
     * @brief Realiza un paso de entrenamiento
     * @return Loss del paso de entrenamiento (0 si no se entrena)
     */
    float train_step() {
        // Verificar si hay suficientes muestras
        if (!replay_buffer->ready(config.min_replay_size)) {
            return 0.0f;
        }
        
        // Solo entrenar cada N pasos
        if (total_steps % config.train_freq != 0) {
            return 0.0f;
        }
        
        // Muestrear batch
        TransitionBatch batch = replay_buffer->sample(config.batch_size);
        
        // Copiar datos a GPU
        CUDA_CHECK(cudaMemcpy(d_states, batch.states.data(),
                              config.batch_size * config.state_size * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_next_states, batch.next_states.data(),
                              config.batch_size * config.state_size * sizeof(float),
                              cudaMemcpyHostToDevice));
        
        // Forward pass de Q-network
        q_network->forward(d_states, d_q_values, config.batch_size);
        
        // Forward pass de target network para next states
        target_network->forward(d_next_states, d_next_q_values, config.batch_size);
        
        // Copiar Q-values a CPU para calcular targets
        std::vector<float> h_q_values(config.batch_size * config.action_size);
        std::vector<float> h_next_q_values(config.batch_size * config.action_size);
        
        CUDA_CHECK(cudaMemcpy(h_q_values.data(), d_q_values,
                              config.batch_size * config.action_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_next_q_values.data(), d_next_q_values,
                              config.batch_size * config.action_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
        
        // Calcular target Q-values y gradientes
        std::vector<float> h_target_q(config.batch_size * config.action_size);
        std::vector<float> h_loss_grad(config.batch_size * config.action_size, 0.0f);
        float batch_loss = 0.0f;
        
        for (int i = 0; i < config.batch_size; i++) {
            int action = batch.actions[i];
            float reward = batch.rewards[i];
            bool done = batch.dones[i];
            
            // Encontrar max Q(s', a') de target network
            float max_next_q = h_next_q_values[i * config.action_size];
            for (int a = 1; a < config.action_size; a++) {
                max_next_q = std::max(max_next_q, 
                                      h_next_q_values[i * config.action_size + a]);
            }
            
            // Calcular target: r + gamma * max_a' Q_target(s', a')
            float target = reward;
            if (!done) {
                target += config.gamma * max_next_q;
            }
            
            // Copiar Q-values actuales como target (solo modificaremos la acción tomada)
            for (int a = 0; a < config.action_size; a++) {
                h_target_q[i * config.action_size + a] = h_q_values[i * config.action_size + a];
            }
            h_target_q[i * config.action_size + action] = target;
            
            // Calcular gradiente del error
            float q_current = h_q_values[i * config.action_size + action];
            float error = q_current - target;
            
            // Gradiente solo para la acción tomada
            h_loss_grad[i * config.action_size + action] = 2.0f * error / config.batch_size;
            
            batch_loss += error * error;
        }
        
        batch_loss /= config.batch_size;
        
        // Copiar gradiente a GPU
        CUDA_CHECK(cudaMemcpy(d_loss_grad, h_loss_grad.data(),
                              config.batch_size * config.action_size * sizeof(float),
                              cudaMemcpyHostToDevice));
        
        // Backward pass
        q_network->backward(d_loss_grad, config.batch_size);
        
        // Actualizar pesos
        train_steps++;
        q_network->update_adam(config.learning_rate, config.adam_beta1,
                               config.adam_beta2, config.adam_epsilon, train_steps);
        
        // Actualizar target network
        if (config.use_soft_update) {
            q_network->soft_update_to(*target_network, config.tau);
        } else if (train_steps % config.target_update_freq == 0) {
            q_network->copy_weights_to(*target_network);
        }
        
        // Actualizar estadísticas
        total_loss += batch_loss;
        loss_count++;
        
        return batch_loss;
    }
    
    /**
     * @brief Llamar al final de cada episodio
     */
    void end_episode() {
        episodes++;
        
        if (config.verbose && episodes % 10 == 0) {
            float avg_loss = (loss_count > 0) ? total_loss / loss_count : 0.0f;
            printf("[DQNAgent] Episodio %d | Steps: %d | Epsilon: %.4f | "
                   "Avg Loss: %.6f | Buffer: %zu\n",
                   episodes, total_steps, epsilon, avg_loss, replay_buffer->size());
            
            // Reset estadísticas
            total_loss = 0.0f;
            loss_count = 0;
        }
    }
    
    /**
     * @brief Guarda el modelo a archivo
     */
    void save(const std::string& filename) {
        q_network->save(filename);
        
        // Guardar también el estado del agente
        std::string state_file = filename + ".state";
        std::ofstream file(state_file, std::ios::binary);
        file.write(reinterpret_cast<char*>(&epsilon), sizeof(float));
        file.write(reinterpret_cast<char*>(&total_steps), sizeof(int));
        file.write(reinterpret_cast<char*>(&train_steps), sizeof(int));
        file.write(reinterpret_cast<char*>(&episodes), sizeof(int));
        printf("[DQNAgent] Estado guardado en: %s\n", state_file.c_str());
    }
    
    /**
     * @brief Carga el modelo desde archivo
     */
    void load(const std::string& filename) {
        q_network->load(filename);
        q_network->copy_weights_to(*target_network);
        
        // Cargar estado del agente si existe
        std::string state_file = filename + ".state";
        std::ifstream file(state_file, std::ios::binary);
        if (file.is_open()) {
            file.read(reinterpret_cast<char*>(&epsilon), sizeof(float));
            file.read(reinterpret_cast<char*>(&total_steps), sizeof(int));
            file.read(reinterpret_cast<char*>(&train_steps), sizeof(int));
            file.read(reinterpret_cast<char*>(&episodes), sizeof(int));
            printf("[DQNAgent] Estado cargado desde: %s\n", state_file.c_str());
        }
    }
    
    // Getters
    float get_epsilon() const { return epsilon; }
    int get_total_steps() const { return total_steps; }
    int get_train_steps() const { return train_steps; }
    int get_episodes() const { return episodes; }
    size_t get_buffer_size() const { return replay_buffer->size(); }
    
    /**
     * @brief Establece epsilon manualmente (para testing)
     */
    void set_epsilon(float eps) { epsilon = eps; }
};

#endif // DQN_AGENT_CUH
