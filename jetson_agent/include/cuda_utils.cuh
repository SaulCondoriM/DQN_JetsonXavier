/**
 * @file cuda_utils.cuh
 * @brief Utilidades CUDA para operaciones de matrices y vectores
 * 
 * Este archivo contiene kernels CUDA optimizados para operaciones
 * de redes neuronales en el Jetson Xavier.
 */

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Macro para verificar errores de CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Macro para verificar errores de cuBLAS
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS Error at %s:%d - status %d\n", \
                    __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Tamaño de bloque para kernels CUDA
constexpr int BLOCK_SIZE = 256;

/**
 * @brief Kernel para activación ReLU
 */
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

/**
 * @brief Kernel para derivada de ReLU (para backprop)
 */
__global__ void relu_backward_kernel(float* grad, const float* activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = (activation[idx] > 0.0f) ? grad[idx] : 0.0f;
    }
}

/**
 * @brief Kernel para sumar bias a cada neurona
 */
__global__ void add_bias_kernel(float* output, const float* bias, 
                                 int batch_size, int neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * neurons;
    
    if (idx < total) {
        int neuron_idx = idx % neurons;
        output[idx] += bias[neuron_idx];
    }
}

/**
 * @brief Kernel para actualización de gradiente de bias
 */
__global__ void update_bias_grad_kernel(float* bias_grad, const float* output_grad,
                                         int batch_size, int neurons) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_idx < neurons) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += output_grad[b * neurons + neuron_idx];
        }
        bias_grad[neuron_idx] = sum;
    }
}

/**
 * @brief Kernel para SGD con momentum
 */
__global__ void sgd_momentum_kernel(float* weights, float* velocity,
                                     const float* gradient,
                                     float learning_rate, float momentum,
                                     int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        velocity[idx] = momentum * velocity[idx] - learning_rate * gradient[idx];
        weights[idx] += velocity[idx];
    }
}

/**
 * @brief Kernel para Adam optimizer
 */
__global__ void adam_update_kernel(float* weights, float* m, float* v,
                                    const float* gradient,
                                    float lr, float beta1, float beta2,
                                    float epsilon, int t, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Actualizar momentos
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * gradient[idx];
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * gradient[idx] * gradient[idx];
        
        // Corrección de sesgo
        float m_hat = m[idx] / (1.0f - powf(beta1, t));
        float v_hat = v[idx] / (1.0f - powf(beta2, t));
        
        // Actualizar pesos
        weights[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

/**
 * @brief Kernel para copiar pesos (soft update para target network)
 */
__global__ void soft_update_kernel(float* target, const float* source,
                                    float tau, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        target[idx] = tau * source[idx] + (1.0f - tau) * target[idx];
    }
}

/**
 * @brief Kernel para calcular error cuadrático medio
 */
__global__ void mse_loss_kernel(float* loss, const float* predicted,
                                 const float* target, int size) {
    __shared__ float partial_sum[BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    float diff = 0.0f;
    if (idx < size) {
        diff = predicted[idx] - target[idx];
        diff = diff * diff;
    }
    partial_sum[tid] = diff;
    __syncthreads();
    
    // Reducción en shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(loss, partial_sum[0] / size);
    }
}

/**
 * @brief Kernel para gradiente de MSE
 */
__global__ void mse_gradient_kernel(float* gradient, const float* predicted,
                                     const float* target, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradient[idx] = 2.0f * (predicted[idx] - target[idx]) / size;
    }
}

/**
 * @brief Kernel para encontrar el índice del máximo (argmax)
 */
__global__ void argmax_kernel(int* result, const float* data, int size) {
    __shared__ float max_vals[BLOCK_SIZE];
    __shared__ int max_idxs[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Inicializar
    max_vals[tid] = -1e30f;
    max_idxs[tid] = 0;
    
    if (idx < size) {
        max_vals[tid] = data[idx];
        max_idxs[tid] = idx;
    }
    __syncthreads();
    
    // Reducción para encontrar máximo
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (max_vals[tid + stride] > max_vals[tid]) {
                max_vals[tid] = max_vals[tid + stride];
                max_idxs[tid] = max_idxs[tid + stride];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *result = max_idxs[0];
    }
}

/**
 * @brief Kernel para obtener el máximo valor
 */
__global__ void max_kernel(float* result, const float* data, int size) {
    __shared__ float max_vals[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    max_vals[tid] = (idx < size) ? data[idx] : -1e30f;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *result = max_vals[0];
    }
}

/**
 * @brief Kernel para inicialización de pesos Xavier
 */
__global__ void xavier_init_kernel(float* weights, int fan_in, int fan_out,
                                    int size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        float scale = sqrtf(2.0f / (fan_in + fan_out));
        weights[idx] = curand_normal(&state) * scale;
    }
}

/**
 * @brief Kernel para generar números aleatorios uniformes
 */
__global__ void uniform_random_kernel(float* data, int size, 
                                       unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = curand_uniform(&state);
    }
}

/**
 * @brief Clase para manejar el contexto CUDA
 */
class CudaContext {
public:
    cublasHandle_t cublas_handle;
    
    CudaContext() {
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        printf("[CUDA] Contexto inicializado\n");
    }
    
    ~CudaContext() {
        cublasDestroy(cublas_handle);
    }
    
    /**
     * Multiplicación de matrices para datos en row-major (como usamos en C++)
     * Calcula: C = A * B
     * Donde:
     *   A tiene dimensiones [M x K]
     *   B tiene dimensiones [K x N]
     *   C tiene dimensiones [M x N]
     * 
     * Para row-major con cuBLAS (que es column-major), usamos el truco:
     * C^T = B^T * A^T
     * Que en cuBLAS se traduce a llamar sgemm con los argumentos intercambiados
     */
    void matmul(const float* A, const float* B, float* C,
                int M, int N, int K,
                float alpha = 1.0f, float beta = 0.0f,
                bool transA = false, bool transB = false) {
        
        // Para row-major storage con cuBLAS:
        // Si no hay transposición: C[M,N] = A[M,K] * B[K,N]
        // En cuBLAS (column-major): C^T = B^T * A^T
        // Llamamos: sgemm(N, M, K, B, N, A, K, C, N)
        
        // Si transA=true: usamos A^T[K,M], así que la "A" efectiva es [K,M]
        // Si transB=true: usamos B^T[N,K], así que la "B" efectiva es [N,K]
        
        cublasOperation_t opA_cublas, opB_cublas;
        int lda, ldb;
        
        // Para cuBLAS, intercambiamos A y B (por el truco row-major -> col-major)
        // Así que transA de nuestra perspectiva afecta a B en cuBLAS y viceversa
        
        if (transB) {
            opA_cublas = CUBLAS_OP_T;
            lda = K;  // B es [K x N], pero transponemos, leading dim es K
        } else {
            opA_cublas = CUBLAS_OP_N;
            lda = N;  // B es [K x N] row-major, leading dim es N
        }
        
        if (transA) {
            opB_cublas = CUBLAS_OP_T;
            ldb = M;  // A es [M x K], pero transponemos, leading dim es M
        } else {
            opB_cublas = CUBLAS_OP_N;
            ldb = K;  // A es [M x K] row-major, leading dim es K
        }
        
        int ldc = N;  // C es [M x N] row-major, leading dim es N
        
        // cuBLAS: C = alpha * op(A) * op(B) + beta * C
        // Pero intercambiamos para row-major: 
        // sgemm(opB_cublas, opA_cublas, N, M, K, alpha, B, lda, A, ldb, beta, C, ldc)
        CUBLAS_CHECK(cublasSgemm(cublas_handle, 
                                  opA_cublas, opB_cublas,
                                  N, M, K,
                                  &alpha, 
                                  B, lda,
                                  A, ldb,
                                  &beta, 
                                  C, ldc));
    }
};

// Funciones helper para lanzar kernels

inline void launch_relu(float* data, int size, cudaStream_t stream = 0) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(data, size);
}

inline void launch_relu_backward(float* grad, const float* activation, 
                                  int size, cudaStream_t stream = 0) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_backward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(grad, activation, size);
}

inline void launch_add_bias(float* output, const float* bias,
                            int batch_size, int neurons, cudaStream_t stream = 0) {
    int total = batch_size * neurons;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_bias_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(output, bias, batch_size, neurons);
}

inline void launch_soft_update(float* target, const float* source,
                               float tau, int size, cudaStream_t stream = 0) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    soft_update_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(target, source, tau, size);
}

inline void launch_adam_update(float* weights, float* m, float* v,
                               const float* gradient, float lr,
                               float beta1, float beta2, float epsilon,
                               int t, int size, cudaStream_t stream = 0) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    adam_update_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        weights, m, v, gradient, lr, beta1, beta2, epsilon, t, size);
}

inline int launch_argmax(const float* data, int size, cudaStream_t stream = 0) {
    int* d_result;
    int h_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
    argmax_kernel<<<1, BLOCK_SIZE, 0, stream>>>(d_result, data, size);
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));
    return h_result;
}

#endif // CUDA_UTILS_CUH
