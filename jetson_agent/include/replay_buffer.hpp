/**
 * @file replay_buffer.hpp
 * @brief Replay Buffer para DQN
 * 
 * Implementación de Experience Replay con muestreo aleatorio
 * para estabilizar el entrenamiento de DQN.
 */

#ifndef REPLAY_BUFFER_HPP
#define REPLAY_BUFFER_HPP

#include <vector>
#include <random>
#include <algorithm>
#include <cstring>

/**
 * @brief Estructura que representa una transición/experiencia
 */
struct Transition {
    std::vector<float> state;
    int action;
    float reward;
    std::vector<float> next_state;
    bool done;
    
    Transition() : action(0), reward(0.0f), done(false) {}
    
    Transition(const std::vector<float>& s, int a, float r,
               const std::vector<float>& ns, bool d)
        : state(s), action(a), reward(r), next_state(ns), done(d) {}
};

/**
 * @brief Batch de transiciones para entrenamiento
 */
struct TransitionBatch {
    std::vector<float> states;      // [batch_size * state_size]
    std::vector<int> actions;       // [batch_size]
    std::vector<float> rewards;     // [batch_size]
    std::vector<float> next_states; // [batch_size * state_size]
    std::vector<bool> dones;        // [batch_size]
    int batch_size;
    int state_size;
    
    TransitionBatch(int bs, int ss) 
        : batch_size(bs), state_size(ss) {
        states.resize(bs * ss);
        actions.resize(bs);
        rewards.resize(bs);
        next_states.resize(bs * ss);
        dones.resize(bs);
    }
};

/**
 * @brief Replay Buffer circular con muestreo aleatorio
 */
class ReplayBuffer {
private:
    std::vector<Transition> buffer;
    size_t capacity;
    size_t position;
    size_t current_size;
    int state_size;
    
    std::mt19937 rng;
    
public:
    /**
     * @brief Constructor
     * @param cap Capacidad máxima del buffer
     * @param state_dim Dimensión del estado
     */
    ReplayBuffer(size_t cap, int state_dim)
        : capacity(cap), position(0), current_size(0), state_size(state_dim) {
        buffer.resize(capacity);
        
        // Inicializar RNG
        std::random_device rd;
        rng.seed(rd());
        
        printf("[ReplayBuffer] Inicializado con capacidad %zu\n", capacity);
    }
    
    /**
     * @brief Añade una transición al buffer
     */
    void add(const std::vector<float>& state, int action, float reward,
             const std::vector<float>& next_state, bool done) {
        buffer[position] = Transition(state, action, reward, next_state, done);
        position = (position + 1) % capacity;
        if (current_size < capacity) {
            current_size++;
        }
    }
    
    /**
     * @brief Añade una transición usando arrays
     */
    void add(const float* state, int action, float reward,
             const float* next_state, bool done) {
        std::vector<float> s(state, state + state_size);
        std::vector<float> ns(next_state, next_state + state_size);
        add(s, action, reward, ns, done);
    }
    
    /**
     * @brief Muestrea un batch aleatorio de transiciones
     * @param batch_size Tamaño del batch
     * @return Batch de transiciones
     */
    TransitionBatch sample(int batch_size) {
        TransitionBatch batch(batch_size, state_size);
        
        // Generar índices aleatorios
        std::uniform_int_distribution<size_t> dist(0, current_size - 1);
        
        for (int i = 0; i < batch_size; i++) {
            size_t idx = dist(rng);
            const Transition& t = buffer[idx];
            
            // Copiar estado
            std::memcpy(&batch.states[i * state_size], 
                       t.state.data(), 
                       state_size * sizeof(float));
            
            // Copiar siguiente estado
            std::memcpy(&batch.next_states[i * state_size],
                       t.next_state.data(),
                       state_size * sizeof(float));
            
            batch.actions[i] = t.action;
            batch.rewards[i] = t.reward;
            batch.dones[i] = t.done;
        }
        
        return batch;
    }
    
    /**
     * @brief Retorna el tamaño actual del buffer
     */
    size_t size() const { return current_size; }
    
    /**
     * @brief Retorna si el buffer tiene suficientes muestras
     */
    bool ready(size_t min_size) const { return current_size >= min_size; }
    
    /**
     * @brief Limpia el buffer
     */
    void clear() {
        position = 0;
        current_size = 0;
    }
    
    /**
     * @brief Retorna estadísticas del buffer
     */
    void print_stats() const {
        printf("[ReplayBuffer] Tamaño: %zu/%zu (%.1f%%)\n",
               current_size, capacity, 
               100.0f * current_size / capacity);
    }
};

#endif // REPLAY_BUFFER_HPP
