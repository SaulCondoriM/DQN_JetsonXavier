/**
 * @file tcp_server.hpp
 * @brief Servidor TCP para recibir estados y enviar acciones
 * 
 * El Jetson actúa como servidor, esperando conexiones del PC simulador.
 */

#ifndef TCP_SERVER_HPP
#define TCP_SERVER_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

/**
 * @brief Información del episodio recibida del simulador
 */
struct EpisodeInfo {
    bool done;
    float reward;
    bool goal_reached;
    bool collision;
};

/**
 * @brief Servidor TCP para comunicación con el simulador
 */
class TCPServer {
private:
    int server_fd;
    int client_fd;
    int port;
    bool connected;
    char buffer[4096];
    std::string recv_buffer;  // Buffer para mensajes parciales
    
public:
    TCPServer(int p = 5555) : server_fd(-1), client_fd(-1), port(p), connected(false), recv_buffer("") {
        memset(buffer, 0, sizeof(buffer));
    }
    
    ~TCPServer() {
        close_connection();
    }
    
    /**
     * @brief Inicia el servidor y espera una conexión
     */
    bool start() {
        // Crear socket
        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) {
            perror("[TCP] Error al crear socket");
            return false;
        }
        
        // Opciones del socket
        int opt = 1;
        if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            perror("[TCP] Error en setsockopt SO_REUSEADDR");
            return false;
        }
        
        // Configurar dirección
        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port);
        
        // Bind
        if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
            perror("[TCP] Error en bind");
            return false;
        }
        
        // Listen
        if (listen(server_fd, 1) < 0) {
            perror("[TCP] Error en listen");
            return false;
        }
        
        printf("[TCP] Servidor escuchando en puerto %d...\n", port);
        printf("[TCP] Esperando conexión del simulador (PC)...\n");
        
        // Accept
        socklen_t addrlen = sizeof(address);
        client_fd = accept(server_fd, (struct sockaddr*)&address, &addrlen);
        if (client_fd < 0) {
            perror("[TCP] Error en accept");
            return false;
        }
        
        // Deshabilitar Nagle para baja latencia
        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
        
        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &address.sin_addr, client_ip, INET_ADDRSTRLEN);
        printf("[TCP] Cliente conectado desde: %s:%d\n", client_ip, ntohs(address.sin_port));
        
        connected = true;
        return true;
    }
    
    /**
     * @brief Cierra la conexión
     */
    void close_connection() {
        if (client_fd >= 0) {
            close(client_fd);
            client_fd = -1;
        }
        if (server_fd >= 0) {
            close(server_fd);
            server_fd = -1;
        }
        connected = false;
        recv_buffer.clear();  // Limpiar buffer al cerrar
        printf("[TCP] Conexión cerrada\n");
    }
    
    /**
     * @brief Intenta parsear un float de forma segura
     * @param str String a parsear
     * @param result Variable donde almacenar el resultado
     * @return true si se parseó correctamente
     */
    bool safe_stof(const std::string& str, float& result) {
        if (str.empty()) return false;
        try {
            // Limpiar espacios en blanco
            size_t start = str.find_first_not_of(" \t\r\n");
            size_t end = str.find_last_not_of(" \t\r\n");
            if (start == std::string::npos) return false;
            
            std::string cleaned = str.substr(start, end - start + 1);
            if (cleaned.empty()) return false;
            
            // Verificar que solo contiene caracteres válidos para un número
            bool has_digit = false;
            for (size_t i = 0; i < cleaned.length(); ++i) {
                char c = cleaned[i];
                if (std::isdigit(c)) {
                    has_digit = true;
                } else if (c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E') {
                    return false;
                }
            }
            if (!has_digit) return false;
            
            result = std::stof(cleaned);
            return true;
        } catch (...) {
            return false;
        }
    }
    
    /**
     * @brief Intenta parsear un int de forma segura
     * @param str String a parsear
     * @param result Variable donde almacenar el resultado
     * @return true si se parseó correctamente
     */
    bool safe_stoi(const std::string& str, int& result) {
        if (str.empty()) return false;
        try {
            size_t start = str.find_first_not_of(" \t\r\n");
            size_t end = str.find_last_not_of(" \t\r\n");
            if (start == std::string::npos) return false;
            
            std::string cleaned = str.substr(start, end - start + 1);
            if (cleaned.empty()) return false;
            
            result = std::stoi(cleaned);
            return true;
        } catch (...) {
            return false;
        }
    }
    
    /**
     * @brief Recibe el estado y la información del episodio
     * @param state Vector para almacenar el estado (10 floats)
     * @param info Información del episodio
     * @return true si se recibió correctamente
     */
    bool receive_state(std::vector<float>& state, EpisodeInfo& info) {
        if (!connected) return false;
        
        // Recibir datos hasta encontrar newline completo
        while (recv_buffer.find('\n') == std::string::npos) {
            memset(buffer, 0, sizeof(buffer));
            int bytes = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
            if (bytes <= 0) {
                if (bytes == 0) {
                    printf("[TCP] Cliente desconectado\n");
                } else {
                    perror("[TCP] Error al recibir datos");
                }
                connected = false;
                return false;
            }
            recv_buffer += std::string(buffer, bytes);
        }
        
        // Extraer primera línea completa
        size_t newline_pos = recv_buffer.find('\n');
        std::string data = recv_buffer.substr(0, newline_pos);
        recv_buffer = recv_buffer.substr(newline_pos + 1);
        
        // Eliminar posible \r al final (Windows)
        if (!data.empty() && data.back() == '\r') {
            data.pop_back();
        }
        
        // Verificar que el mensaje no esté vacío
        if (data.empty()) {
            printf("[TCP] Mensaje vacío recibido, ignorando\n");
            return false;
        }
        
        // Parsear: estado|done,reward,goal,collision
        size_t pipe_pos = data.find('|');
        if (pipe_pos == std::string::npos) {
            printf("[TCP] Formato de mensaje inválido (sin separador '|'): '%s'\n", 
                   data.substr(0, 50).c_str());
            return false;
        }
        
        std::string state_str = data.substr(0, pipe_pos);
        std::string info_str = data.substr(pipe_pos + 1);
        
        // Parsear estado (10 valores separados por comas)
        state.clear();
        std::stringstream ss(state_str);
        std::string token;
        int token_count = 0;
        bool parse_error = false;
        
        while (std::getline(ss, token, ',')) {
            token_count++;
            float value;
            if (safe_stof(token, value)) {
                state.push_back(value);
            } else {
                printf("[TCP] Error parseando token %d: '%s'\n", token_count, token.c_str());
                parse_error = true;
                break;
            }
        }
        
        if (parse_error || state.size() != 10) {
            printf("[TCP] Error: se esperaban 10 valores de estado, se recibieron %zu\n",
                   state.size());
            printf("[TCP] Mensaje problemático: '%s'\n", data.substr(0, 100).c_str());
            state.clear();
            return false;
        }
        
        // Parsear info del episodio
        std::stringstream ss_info(info_str);
        std::vector<std::string> info_parts;
        while (std::getline(ss_info, token, ',')) {
            info_parts.push_back(token);
        }
        
        if (info_parts.size() >= 4) {
            int done_val, goal_val, collision_val;
            float reward_val;
            
            if (safe_stoi(info_parts[0], done_val) &&
                safe_stof(info_parts[1], reward_val) &&
                safe_stoi(info_parts[2], goal_val) &&
                safe_stoi(info_parts[3], collision_val)) {
                
                info.done = (done_val != 0);
                info.reward = reward_val;
                info.goal_reached = (goal_val != 0);
                info.collision = (collision_val != 0);
            } else {
                printf("[TCP] Error parseando info del episodio: '%s'\n", info_str.c_str());
                return false;
            }
        } else {
            printf("[TCP] Info del episodio incompleta: '%s'\n", info_str.c_str());
            return false;
        }
        
        return true;
    }
    
    /**
     * @brief Envía la acción al simulador
     * @param action Índice de la acción (0-4)
     * @return true si se envió correctamente
     */
    bool send_action(int action) {
        if (!connected) return false;
        
        std::string msg = std::to_string(action) + "\n";
        
        int bytes = send(client_fd, msg.c_str(), msg.length(), 0);
        if (bytes <= 0) {
            perror("[TCP] Error al enviar acción");
            connected = false;
            return false;
        }
        
        return true;
    }
    
    /**
     * @brief Verifica si está conectado
     */
    bool is_connected() const { return connected; }
};

#endif // TCP_SERVER_HPP
