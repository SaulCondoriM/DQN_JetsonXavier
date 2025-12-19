# Makefile para compilar el agente DQN en Jetson Xavier
# 
# Uso:
#   make          - Compila el programa
#   make clean    - Limpia archivos compilados
#   make run      - Ejecuta el programa
#   make debug    - Compila con información de depuración

# Compilador CUDA
NVCC = nvcc

# Flags de compilación
NVCC_FLAGS = -std=c++14 -O3 -arch=sm_72
NVCC_FLAGS += -Xcompiler -Wall,-Wextra

# Flags para Jetson Xavier (sm_72)
# Para otras arquitecturas:
#   Jetson Nano:   sm_53
#   Jetson TX2:    sm_62
#   Jetson Xavier: sm_72
#   Jetson Orin:   sm_87

# Librerías
LIBS = -lcublas -lcurand

# Directorios
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Archivos fuente
SOURCES = $(SRC_DIR)/main.cu
HEADERS = $(wildcard $(INC_DIR)/*.cuh) $(wildcard $(INC_DIR)/*.hpp)

# Ejecutable
TARGET = $(BIN_DIR)/dqn_agent

# Reglas
.PHONY: all clean run debug

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(BUILD_DIR)
	@echo "Compilando DQN Agent..."
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) $(SOURCES) -o $(TARGET) $(LIBS)
	@echo "Compilación exitosa: $(TARGET)"

debug: NVCC_FLAGS += -g -G -DDEBUG
debug: $(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "Limpieza completada"

run: $(TARGET)
	./$(TARGET)

# Regla para verificar la arquitectura CUDA
check-cuda:
	@echo "Verificando dispositivo CUDA..."
	@nvcc --version
	@nvidia-smi

# Instalar dependencias (para sistemas basados en apt)
install-deps:
	sudo apt-get update
	sudo apt-get install -y build-essential
	@echo "Asegúrate de tener CUDA Toolkit instalado"

# Generar documentación
docs:
	@echo "Generando documentación..."
	doxygen Doxyfile 2>/dev/null || echo "Instala doxygen para generar documentación"

help:
	@echo "Makefile para DQN Agent - Jetson Xavier"
	@echo ""
	@echo "Targets disponibles:"
	@echo "  all          - Compila el programa (default)"
	@echo "  debug        - Compila con información de depuración"
	@echo "  clean        - Elimina archivos compilados"
	@echo "  run          - Ejecuta el programa"
	@echo "  check-cuda   - Verifica el dispositivo CUDA"
	@echo "  install-deps - Instala dependencias básicas"
	@echo "  help         - Muestra esta ayuda"
