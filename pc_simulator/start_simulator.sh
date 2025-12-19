#!/bin/bash
#
# Script para iniciar el simulador en el PC
# Uso: ./start_simulator.sh <IP_JETSON> [opciones]
#

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Valores por defecto
JETSON_IP="${1:-127.0.0.1}"
PORT="${2:-5555}"
EPISODES="${3:-1000}"
VISUALIZE="${4:-true}"

# Directorio del script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}    Simulador de Robot Diferencial - DQN   ${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "Jetson IP:   ${YELLOW}$JETSON_IP${NC}"
echo -e "Puerto:      ${YELLOW}$PORT${NC}"
echo -e "Episodios:   ${YELLOW}$EPISODES${NC}"
echo -e "Visualizar:  ${YELLOW}$VISUALIZE${NC}"
echo ""

# Verificar conectividad
echo -e "${YELLOW}Verificando conectividad con el Jetson...${NC}"
if ping -c 1 -W 2 "$JETSON_IP" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Conexión establecida${NC}"
else
    echo -e "${RED}✗ No se puede conectar a $JETSON_IP${NC}"
    echo -e "${YELLOW}Asegúrate de que el Jetson esté encendido y conectado${NC}"
    read -p "¿Continuar de todos modos? (s/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        exit 1
    fi
fi

# Construir comando
CMD="python3 ${SCRIPT_DIR}/tcp_client.py --ip $JETSON_IP --port $PORT --episodes $EPISODES"

if [ "$VISUALIZE" = "true" ]; then
    CMD="$CMD --visualize"
fi

echo ""
echo -e "${YELLOW}Iniciando simulador...${NC}"
echo -e "Comando: ${CMD}"
echo ""

# Ejecutar
exec $CMD
