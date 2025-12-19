#!/bin/bash
#
# Script para iniciar el agente DQN en el Jetson Xavier
# Uso: ./start_agent.sh [puerto] [episodios]
#

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PORT="${1:-5555}"
EPISODES="${2:--1}"
MODEL_PATH="${3:-models/dqn_model.bin}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}    Agente DQN - Jetson Xavier             ${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Verificar CUDA
echo -e "${YELLOW}Verificando CUDA...${NC}"
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}✗ NVCC no encontrado. Verifica la instalación de CUDA${NC}"
    exit 1
fi
echo -e "${GREEN}✓ CUDA disponible$(nvcc --version | grep release | sed 's/.*release //')${NC}"

# Verificar GPU
echo -e "${YELLOW}Verificando GPU...${NC}"
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi no disponible${NC}"
    exit 1
fi
echo -e "${GREEN}✓ GPU detectada${NC}"

# Compilar si es necesario
if [ ! -f "bin/dqn_agent" ] || [ "src/main.cu" -nt "bin/dqn_agent" ]; then
    echo ""
    echo -e "${YELLOW}Compilando agente...${NC}"
    make clean && make
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Error de compilación${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Compilación exitosa${NC}"
fi

# Crear directorio de modelos
mkdir -p models

# Mostrar IP local
echo ""
echo -e "${YELLOW}Direcciones IP disponibles:${NC}"
ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | while read ip; do
    echo -e "  - ${GREEN}$ip${NC}"
done

echo ""
echo -e "Puerto:      ${YELLOW}$PORT${NC}"
echo -e "Episodios:   ${YELLOW}$EPISODES${NC} (-1 = infinito)"
echo -e "Modelo:      ${YELLOW}$MODEL_PATH${NC}"
echo ""
echo -e "${YELLOW}Esperando conexión del simulador (PC)...${NC}"
echo -e "${YELLOW}Presiona Ctrl+C para detener${NC}"
echo ""

# Ejecutar agente
exec ./bin/dqn_agent --port "$PORT" --episodes "$EPISODES" --model "$MODEL_PATH"
