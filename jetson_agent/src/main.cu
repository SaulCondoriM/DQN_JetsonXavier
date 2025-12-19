/**
 * @file main.cu
 * @brief Programa principal del agente DQN para Jetson Xavier
 * 
 * Este programa:
 * 1. Inicia un servidor TCP
 * 2. Recibe estados del simulador (PC)
 * 3. Ejecuta inferencia DQN para seleccionar acciones
 * 4. Envía acciones al simulador
 * 5. Entrena el modelo online con las transiciones
 */

#include <iostream>
#include <csignal>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <iomanip>

#include "../include/dqn_agent.cuh"
#include "../include/tcp_server.hpp"

// Flag para detener el programa limpiamente
volatile bool running = true;

void signal_handler(int signum) {
    printf("\n[Main] Señal %d recibida, deteniendo...\n", signum);
    running = false;
}

/**
 * @brief Imprime estadísticas del episodio
 */
void print_episode_stats(int episode, int steps, float total_reward,
                         bool goal_reached, bool collision, float epsilon,
                         double episode_time) {
    std::string result = goal_reached ? "✓ GOAL" : (collision ? "✗ COLLISION" : "- TIMEOUT");
    
    printf("Ep %4d | Steps: %4d | Reward: %8.2f | %s | ε: %.4f | Time: %.2fs\n",
           episode, steps, total_reward, result.c_str(), epsilon, episode_time);
}

/**
 * @brief Loop principal del agente
 */
void run_agent(DQNAgent& agent, TCPServer& server, int max_episodes = -1,
               const std::string& save_path = "models/dqn_model.bin") {
    
    printf("\n========================================\n");
    printf("   DQN Agent - Jetson Xavier\n");
    printf("========================================\n\n");
    
    std::vector<float> state(10);
    std::vector<float> prev_state(10);
    EpisodeInfo info;
    
    int episode = 0;
    int total_steps = 0;
    int total_goals = 0;
    int total_collisions = 0;
    
    float best_reward = -1e9;
    std::vector<float> episode_rewards;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (running && (max_episodes < 0 || episode < max_episodes)) {
        // Esperar primer estado del episodio
        if (!server.receive_state(state, info)) {
            printf("[Main] Error al recibir estado inicial\n");
            break;
        }
        
        auto episode_start = std::chrono::high_resolution_clock::now();
        
        episode++;
        int episode_steps = 0;
        float episode_reward = 0.0f;
        bool episode_done = false;
        
        // Seleccionar primera acción
        int action = agent.select_action(state.data());
        
        // Enviar acción
        if (!server.send_action(action)) {
            printf("[Main] Error al enviar acción\n");
            break;
        }
        
        while (running && !episode_done) {
            // Guardar estado actual
            prev_state = state;
            int prev_action = action;
            
            // Recibir nuevo estado
            if (!server.receive_state(state, info)) {
                printf("[Main] Error al recibir estado\n");
                break;
            }
            
            episode_steps++;
            total_steps++;
            episode_reward += info.reward;
            episode_done = info.done;
            
            // Almacenar transición
            agent.store_transition(prev_state.data(), prev_action, info.reward,
                                   state.data(), info.done);
            
            // Entrenar
            agent.train_step();
            
            // Seleccionar siguiente acción
            action = agent.select_action(state.data());
            
            // Enviar acción
            if (!server.send_action(action)) {
                printf("[Main] Error al enviar acción\n");
                running = false;
                break;
            }
        }
        
        auto episode_end = std::chrono::high_resolution_clock::now();
        double episode_time = std::chrono::duration<double>(
            episode_end - episode_start).count();
        
        // Finalizar episodio
        agent.end_episode();
        
        // Actualizar estadísticas
        if (info.goal_reached) total_goals++;
        if (info.collision) total_collisions++;
        episode_rewards.push_back(episode_reward);
        
        // Imprimir estadísticas
        print_episode_stats(episode, episode_steps, episode_reward,
                           info.goal_reached, info.collision,
                           agent.get_epsilon(), episode_time);
        
        // Guardar mejor modelo
        if (episode_reward > best_reward) {
            best_reward = episode_reward;
            agent.save(save_path);
            printf("  └── Nuevo mejor modelo guardado (reward: %.2f)\n", best_reward);
        }
        
        // Guardar periódicamente
        if (episode % 100 == 0) {
            agent.save(save_path + ".checkpoint");
            
            // Calcular estadísticas
            float avg_reward = 0.0f;
            int window = std::min(100, (int)episode_rewards.size());
            for (int i = episode_rewards.size() - window; i < (int)episode_rewards.size(); i++) {
                avg_reward += episode_rewards[i];
            }
            avg_reward /= window;
            
            auto current_time = std::chrono::high_resolution_clock::now();
            double total_time = std::chrono::duration<double>(
                current_time - start_time).count();
            
            printf("\n=== Checkpoint (Episodio %d) ===\n", episode);
            printf("  Tiempo total: %.1f min\n", total_time / 60.0);
            printf("  Steps totales: %d\n", total_steps);
            printf("  Objetivos alcanzados: %d (%.1f%%)\n", 
                   total_goals, 100.0f * total_goals / episode);
            printf("  Colisiones: %d (%.1f%%)\n",
                   total_collisions, 100.0f * total_collisions / episode);
            printf("  Reward promedio (últimos 100): %.2f\n", avg_reward);
            printf("  Mejor reward: %.2f\n", best_reward);
            printf("================================\n\n");
        }
    }
    
    // Resumen final
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    printf("\n========== RESUMEN FINAL ==========\n");
    printf("Episodios completados: %d\n", episode);
    printf("Steps totales: %d\n", total_steps);
    printf("Tiempo total: %.1f min\n", total_time / 60.0);
    printf("Objetivos alcanzados: %d (%.1f%%)\n",
           total_goals, 100.0f * total_goals / episode);
    printf("Colisiones: %d (%.1f%%)\n",
           total_collisions, 100.0f * total_collisions / episode);
    
    if (!episode_rewards.empty()) {
        float avg = 0.0f;
        for (float r : episode_rewards) avg += r;
        avg /= episode_rewards.size();
        printf("Reward promedio: %.2f\n", avg);
    }
    printf("Mejor reward: %.2f\n", best_reward);
    printf("===================================\n");
    
    // Guardar modelo final
    agent.save(save_path + ".final");
}

/**
 * @brief Función principal
 */
int main(int argc, char** argv) {
    // Configurar signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Parsear argumentos
    int port = 5555;
    int max_episodes = -1;
    std::string model_path = "models/dqn_model.bin";
    bool load_model = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--episodes" && i + 1 < argc) {
            max_episodes = std::atoi(argv[++i]);
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--load") {
            load_model = true;
        } else if (arg == "--help") {
            printf("Uso: %s [opciones]\n", argv[0]);
            printf("Opciones:\n");
            printf("  --port N        Puerto TCP (default: 5555)\n");
            printf("  --episodes N    Número máximo de episodios (-1 = infinito)\n");
            printf("  --model PATH    Ruta para guardar/cargar modelo\n");
            printf("  --load          Cargar modelo existente\n");
            printf("  --help          Mostrar esta ayuda\n");
            return 0;
        }
    }
    
    // Verificar CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("ERROR: No se encontró dispositivo CUDA\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("[CUDA] Dispositivo: %s\n", prop.name);
    printf("[CUDA] Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("[CUDA] Memoria global: %.0f MB\n", prop.totalGlobalMem / 1e6);
    
    // Configuración del agente
    DQNConfig config;
    config.state_size = 10;
    config.action_size = 5;
    config.hidden_sizes = {128, 128, 64};
    config.learning_rate = 0.0005f;
    config.gamma = 0.99f;
    config.batch_size = 64;
    config.replay_buffer_size = 100000;
    config.min_replay_size = 500;  // Empezar a entrenar más rápido
    config.epsilon_start = 1.0f;
    config.epsilon_end = 0.05f;   // No bajar tanto el epsilon
    // Con 0.9999, después de 10000 pasos: epsilon = 0.37
    // Con 0.99995, después de 10000 pasos: epsilon = 0.61
    config.epsilon_decay = 0.9999f;
    config.tau = 0.001f;  // Soft update más suave
    config.target_update_freq = 100;
    config.train_freq = 1;  // Entrenar en cada paso
    config.verbose = false;
    
    printf("\n[Config] DQN Agent\n");
    printf("  State size: %d\n", config.state_size);
    printf("  Action size: %d\n", config.action_size);
    printf("  Hidden layers: ");
    for (int h : config.hidden_sizes) printf("%d ", h);
    printf("\n");
    printf("  Learning rate: %.6f\n", config.learning_rate);
    printf("  Gamma: %.4f\n", config.gamma);
    printf("  Batch size: %d\n", config.batch_size);
    printf("  Replay buffer: %d\n", config.replay_buffer_size);
    printf("  Epsilon: %.2f -> %.2f (decay: %.4f)\n",
           config.epsilon_start, config.epsilon_end, config.epsilon_decay);
    
    // Crear agente
    DQNAgent agent(config);
    
    // Cargar modelo si se especificó
    if (load_model) {
        try {
            agent.load(model_path);
            printf("[Main] Modelo cargado desde: %s\n", model_path.c_str());
        } catch (const std::exception& e) {
            printf("[Main] No se pudo cargar modelo: %s\n", e.what());
            printf("[Main] Continuando con modelo nuevo\n");
        }
    }
    
    // Crear directorio para modelos
    system("mkdir -p models");
    
    // Iniciar servidor TCP
    TCPServer server(port);
    if (!server.start()) {
        printf("ERROR: No se pudo iniciar el servidor TCP\n");
        return 1;
    }
    
    // Ejecutar agente
    run_agent(agent, server, max_episodes, model_path);
    
    // Cerrar servidor
    server.close_connection();
    
    printf("[Main] Programa terminado\n");
    return 0;
}
