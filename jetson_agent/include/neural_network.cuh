/**
 * @file neural_network.cuh
 * @brief Red neuronal fully-connected con soporte CUDA
 * 
 * Implementación de una red neuronal profunda para DQN
 * optimizada para Jetson Xavier usando CUDA.
 */

#ifndef NEURAL_NETWORK_CUH
#define NEURAL_NETWORK_CUH

#include "cuda_utils.cuh"
#include <vector>
#include <random>
#include <cstring>
#include <fstream>

/**
 * @brief Capa fully-connected con activación ReLU opcional
 */
class DenseLayer {
public:
    int input_size;
    int output_size;
    bool use_relu;
    
    // Pesos y bias en GPU
    float* d_weights;
    float* d_bias;
    
    // Gradientes
    float* d_weights_grad;
    float* d_bias_grad;
    
    // Momentos para Adam
    float* d_weights_m;
    float* d_weights_v;
    float* d_bias_m;
    float* d_bias_v;
    
    // Activaciones para backprop
    float* d_input_cache;
    float* d_output_cache;
    float* d_pre_activation;
    
    int batch_size;
    
    DenseLayer(int in_size, int out_size, bool relu = true, int max_batch = 64)
        : input_size(in_size), output_size(out_size), use_relu(relu), batch_size(max_batch) {
        
        int weights_size = input_size * output_size;
        
        // Allocar memoria en GPU
        CUDA_CHECK(cudaMalloc(&d_weights, weights_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias, output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_grad, weights_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias_grad, output_size * sizeof(float)));
        
        // Momentos Adam
        CUDA_CHECK(cudaMalloc(&d_weights_m, weights_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_v, weights_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias_m, output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias_v, output_size * sizeof(float)));
        
        // Cache para backprop
        CUDA_CHECK(cudaMalloc(&d_input_cache, batch_size * input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_cache, batch_size * output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pre_activation, batch_size * output_size * sizeof(float)));
        
        // Inicializar momentos a cero
        CUDA_CHECK(cudaMemset(d_weights_m, 0, weights_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_weights_v, 0, weights_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_bias_m, 0, output_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_bias_v, 0, output_size * sizeof(float)));
        
        // Inicializar pesos con Xavier
        initialize_weights();
    }
    
    ~DenseLayer() {
        cudaFree(d_weights);
        cudaFree(d_bias);
        cudaFree(d_weights_grad);
        cudaFree(d_bias_grad);
        cudaFree(d_weights_m);
        cudaFree(d_weights_v);
        cudaFree(d_bias_m);
        cudaFree(d_bias_v);
        cudaFree(d_input_cache);
        cudaFree(d_output_cache);
        cudaFree(d_pre_activation);
    }
    
    void initialize_weights() {
        int size = input_size * output_size;
        
        // Xavier initialization en CPU y copiar a GPU
        std::random_device rd;
        std::mt19937 gen(rd());
        float scale = std::sqrt(2.0f / (input_size + output_size));
        std::normal_distribution<float> dist(0.0f, scale);
        
        std::vector<float> h_weights(size);
        std::vector<float> h_bias(output_size, 0.0f);
        
        for (int i = 0; i < size; i++) {
            h_weights[i] = dist(gen);
        }
        
        CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), 
                              size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), 
                              output_size * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    /**
     * @brief Forward pass
     * @param input Entrada [batch_size x input_size]
     * @param output Salida [batch_size x output_size]
     * @param ctx Contexto CUDA con cuBLAS handle
     * @param current_batch Tamaño del batch actual
     * 
     * Cálculo: output = input * weights + bias
     * Donde weights tiene forma [input_size x output_size]
     */
    void forward(const float* input, float* output, CudaContext& ctx, int current_batch) {
        // Guardar input para backprop
        CUDA_CHECK(cudaMemcpy(d_input_cache, input, 
                              current_batch * input_size * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        
        // output = input * weights
        // input: [batch_size x input_size]
        // weights: [input_size x output_size]  
        // output: [batch_size x output_size]
        ctx.matmul(input, d_weights, output,
                   current_batch, output_size, input_size,
                   1.0f, 0.0f, false, false);
        
        // Añadir bias
        launch_add_bias(output, d_bias, current_batch, output_size);
        
        // Guardar pre-activación
        CUDA_CHECK(cudaMemcpy(d_pre_activation, output,
                              current_batch * output_size * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        
        // Aplicar ReLU si corresponde
        if (use_relu) {
            launch_relu(output, current_batch * output_size);
        }
        
        // Guardar output para backprop
        CUDA_CHECK(cudaMemcpy(d_output_cache, output,
                              current_batch * output_size * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }
    
    /**
     * @brief Backward pass
     * @param output_grad Gradiente de la salida [batch_size x output_size]
     * @param input_grad Gradiente de la entrada [batch_size x input_size]
     * @param ctx Contexto CUDA
     * @param current_batch Tamaño del batch actual
     */
    void backward(const float* output_grad, float* input_grad, 
                  CudaContext& ctx, int current_batch) {
        // Crear copia del gradiente para modificar
        float* d_grad_relu;
        CUDA_CHECK(cudaMalloc(&d_grad_relu, 
                              current_batch * output_size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_grad_relu, output_grad,
                              current_batch * output_size * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        
        // Aplicar gradiente de ReLU
        if (use_relu) {
            launch_relu_backward(d_grad_relu, d_pre_activation, 
                                current_batch * output_size);
        }
        
        // Calcular gradiente de pesos: weights_grad = input^T * grad_relu
        // input_cache: [batch x input_size]
        // d_grad_relu: [batch x output_size]
        // weights_grad: [input_size x output_size]
        // Necesitamos: input^T * grad = [input_size x batch] * [batch x output_size]
        ctx.matmul(d_input_cache, d_grad_relu, d_weights_grad,
                   input_size, output_size, current_batch,
                   1.0f, 0.0f, true, false);
        
        // Calcular gradiente de bias
        int blocks = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        update_bias_grad_kernel<<<blocks, BLOCK_SIZE>>>(
            d_bias_grad, d_grad_relu, current_batch, output_size);
        
        // Calcular gradiente de entrada si es necesario
        if (input_grad != nullptr) {
            // input_grad = grad_relu * weights^T
            // d_grad_relu: [batch x output_size]
            // weights: [input_size x output_size]
            // input_grad: [batch x input_size]
            // Necesitamos: grad * W^T = [batch x output_size] * [output_size x input_size]
            ctx.matmul(d_grad_relu, d_weights, input_grad,
                       current_batch, input_size, output_size,
                       1.0f, 0.0f, false, true);
        }
        
        cudaFree(d_grad_relu);
    }
    
    /**
     * @brief Actualiza pesos con Adam optimizer
     */
    void update_adam(float lr, float beta1, float beta2, float epsilon, int t) {
        int weights_size = input_size * output_size;
        
        launch_adam_update(d_weights, d_weights_m, d_weights_v, d_weights_grad,
                          lr, beta1, beta2, epsilon, t, weights_size);
        
        launch_adam_update(d_bias, d_bias_m, d_bias_v, d_bias_grad,
                          lr, beta1, beta2, epsilon, t, output_size);
    }
    
    /**
     * @brief Copia pesos a otra capa (para target network)
     */
    void copy_weights_to(DenseLayer& target) {
        int weights_size = input_size * output_size;
        CUDA_CHECK(cudaMemcpy(target.d_weights, d_weights,
                              weights_size * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(target.d_bias, d_bias,
                              output_size * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }
    
    /**
     * @brief Soft update de pesos (para target network)
     */
    void soft_update_to(DenseLayer& target, float tau) {
        int weights_size = input_size * output_size;
        launch_soft_update(target.d_weights, d_weights, tau, weights_size);
        launch_soft_update(target.d_bias, d_bias, tau, output_size);
    }
    
    /**
     * @brief Guarda pesos a archivo
     */
    void save(std::ofstream& file) {
        int weights_size = input_size * output_size;
        std::vector<float> h_weights(weights_size);
        std::vector<float> h_bias(output_size);
        
        CUDA_CHECK(cudaMemcpy(h_weights.data(), d_weights,
                              weights_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_bias.data(), d_bias,
                              output_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
        
        file.write(reinterpret_cast<char*>(&input_size), sizeof(int));
        file.write(reinterpret_cast<char*>(&output_size), sizeof(int));
        file.write(reinterpret_cast<char*>(h_weights.data()), 
                   weights_size * sizeof(float));
        file.write(reinterpret_cast<char*>(h_bias.data()), 
                   output_size * sizeof(float));
    }
    
    /**
     * @brief Carga pesos desde archivo
     */
    void load(std::ifstream& file) {
        int in_size, out_size;
        file.read(reinterpret_cast<char*>(&in_size), sizeof(int));
        file.read(reinterpret_cast<char*>(&out_size), sizeof(int));
        
        if (in_size != input_size || out_size != output_size) {
            throw std::runtime_error("Layer size mismatch");
        }
        
        int weights_size = input_size * output_size;
        std::vector<float> h_weights(weights_size);
        std::vector<float> h_bias(output_size);
        
        file.read(reinterpret_cast<char*>(h_weights.data()),
                  weights_size * sizeof(float));
        file.read(reinterpret_cast<char*>(h_bias.data()),
                  output_size * sizeof(float));
        
        CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(),
                              weights_size * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(),
                              output_size * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
};


/**
 * @brief Red neuronal profunda para DQN
 */
class DQNNetwork {
public:
    std::vector<DenseLayer*> layers;
    CudaContext ctx;
    
    int input_size;
    int output_size;
    int max_batch_size;
    
    // Buffers intermedios
    std::vector<float*> d_layer_outputs;
    float* d_input;
    float* d_output;
    
    /**
     * @brief Constructor
     * @param state_size Tamaño del estado (entrada)
     * @param action_size Número de acciones (salida)
     * @param hidden_sizes Vector con tamaños de capas ocultas
     * @param max_batch Tamaño máximo de batch
     */
    DQNNetwork(int state_size, int action_size, 
               const std::vector<int>& hidden_sizes = {128, 128},
               int max_batch = 64)
        : input_size(state_size), output_size(action_size), 
          max_batch_size(max_batch) {
        
        // Crear capas
        int prev_size = state_size;
        for (int hidden : hidden_sizes) {
            layers.push_back(new DenseLayer(prev_size, hidden, true, max_batch));
            prev_size = hidden;
        }
        // Capa de salida sin ReLU
        layers.push_back(new DenseLayer(prev_size, action_size, false, max_batch));
        
        // Allocar buffers
        CUDA_CHECK(cudaMalloc(&d_input, max_batch * state_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, max_batch * action_size * sizeof(float)));
        
        for (size_t i = 0; i < layers.size(); i++) {
            float* buffer;
            int size = max_batch * layers[i]->output_size;
            CUDA_CHECK(cudaMalloc(&buffer, size * sizeof(float)));
            d_layer_outputs.push_back(buffer);
        }
        
        printf("[DQN] Red creada: %d -> ", state_size);
        for (int h : hidden_sizes) printf("%d -> ", h);
        printf("%d\n", action_size);
    }
    
    ~DQNNetwork() {
        for (auto layer : layers) delete layer;
        for (auto buffer : d_layer_outputs) cudaFree(buffer);
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    /**
     * @brief Forward pass completo
     * @param state Estado de entrada [batch_size x input_size]
     * @param q_values Q-values de salida [batch_size x output_size]
     * @param batch_size Tamaño del batch
     */
    void forward(const float* state, float* q_values, int batch_size = 1) {
        // Copiar entrada a GPU si es necesario
        const float* current_input = state;
        
        for (size_t i = 0; i < layers.size(); i++) {
            float* current_output = d_layer_outputs[i];
            layers[i]->forward(current_input, current_output, ctx, batch_size);
            current_input = current_output;
        }
        
        // Copiar salida
        CUDA_CHECK(cudaMemcpy(q_values, d_layer_outputs.back(),
                              batch_size * output_size * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }
    
    /**
     * @brief Forward desde CPU
     */
    void forward_cpu(const float* h_state, float* h_q_values, int batch_size = 1) {
        // Copiar estado a GPU
        CUDA_CHECK(cudaMemcpy(d_input, h_state,
                              batch_size * input_size * sizeof(float),
                              cudaMemcpyHostToDevice));
        
        // Forward
        forward(d_input, d_output, batch_size);
        
        // Copiar resultado a CPU
        CUDA_CHECK(cudaMemcpy(h_q_values, d_output,
                              batch_size * output_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    
    /**
     * @brief Backward pass completo
     * @param output_grad Gradiente de la salida
     * @param batch_size Tamaño del batch
     */
    void backward(const float* output_grad, int batch_size) {
        const float* current_grad = output_grad;
        
        for (int i = layers.size() - 1; i >= 0; i--) {
            float* input_grad = (i > 0) ? d_layer_outputs[i-1] : nullptr;
            
            // Reusar buffer para gradiente de entrada
            float* d_temp_grad;
            if (i > 0) {
                int prev_size = layers[i-1]->output_size;
                CUDA_CHECK(cudaMalloc(&d_temp_grad, batch_size * prev_size * sizeof(float)));
            } else {
                d_temp_grad = nullptr;
            }
            
            layers[i]->backward(current_grad, d_temp_grad, ctx, batch_size);
            
            if (i > 0) {
                // Copiar gradiente para la siguiente iteración
                CUDA_CHECK(cudaMemcpy(d_layer_outputs[i-1], d_temp_grad,
                                      batch_size * layers[i-1]->output_size * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
                cudaFree(d_temp_grad);
                current_grad = d_layer_outputs[i-1];
            }
        }
    }
    
    /**
     * @brief Actualiza pesos con Adam
     */
    void update_adam(float lr, float beta1, float beta2, float epsilon, int t) {
        for (auto layer : layers) {
            layer->update_adam(lr, beta1, beta2, epsilon, t);
        }
    }
    
    /**
     * @brief Copia pesos a otra red (target network)
     */
    void copy_weights_to(DQNNetwork& target) {
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i]->copy_weights_to(*target.layers[i]);
        }
    }
    
    /**
     * @brief Soft update a target network
     */
    void soft_update_to(DQNNetwork& target, float tau) {
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i]->soft_update_to(*target.layers[i], tau);
        }
    }
    
    /**
     * @brief Obtiene la mejor acción para un estado
     */
    int get_best_action(const float* h_state) {
        std::vector<float> q_values(output_size);
        forward_cpu(h_state, q_values.data(), 1);
        
        int best_action = 0;
        float best_q = q_values[0];
        for (int i = 1; i < output_size; i++) {
            if (q_values[i] > best_q) {
                best_q = q_values[i];
                best_action = i;
            }
        }
        return best_action;
    }
    
    /**
     * @brief Guarda la red a archivo
     */
    void save(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }
        
        int num_layers = layers.size();
        file.write(reinterpret_cast<char*>(&num_layers), sizeof(int));
        
        for (auto layer : layers) {
            layer->save(file);
        }
        
        printf("[DQN] Modelo guardado en: %s\n", filename.c_str());
    }
    
    /**
     * @brief Carga la red desde archivo
     */
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for reading: " + filename);
        }
        
        int num_layers;
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
        
        if (num_layers != (int)layers.size()) {
            throw std::runtime_error("Layer count mismatch");
        }
        
        for (auto layer : layers) {
            layer->load(file);
        }
        
        printf("[DQN] Modelo cargado desde: %s\n", filename.c_str());
    }
};

#endif // NEURAL_NETWORK_CUH
