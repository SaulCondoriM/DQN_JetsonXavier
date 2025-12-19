#!/usr/bin/env python3
"""
Script de prueba de comunicación TCP
Ejecuta el servidor y el cliente en modo de prueba
"""

import subprocess
import sys
import time
import os

def test_basic_communication():
    """Prueba básica de comunicación TCP"""
    print("="*60)
    print("  TEST DE COMUNICACIÓN TCP BÁSICO")
    print("="*60)
    print()
    
    # Cambiar al directorio correcto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("[1] Iniciando servidor de prueba en segundo plano...")
    server_proc = subprocess.Popen(
        [sys.executable, "test_server.py", "--port", "5555", "--episodes", "5"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    # Esperar a que el servidor esté listo
    time.sleep(2)
    
    print("[2] Iniciando cliente (simulador)...")
    client_proc = subprocess.Popen(
        [sys.executable, "tcp_client.py", "--ip", "127.0.0.1", "--port", "5555", "--episodes", "5"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    print("[3] Esperando resultados...")
    print("-"*60)
    
    # Esperar a que terminen
    try:
        stdout, _ = client_proc.communicate(timeout=120)
        print("\n--- Salida del Cliente ---")
        print(stdout.decode('utf-8'))
        
        server_proc.terminate()
        stdout, _ = server_proc.communicate(timeout=5)
        print("\n--- Salida del Servidor ---")
        print(stdout.decode('utf-8'))
        
        print("-"*60)
        print("✓ Test completado exitosamente!")
        return True
        
    except subprocess.TimeoutExpired:
        print("✗ Timeout - los procesos tardaron demasiado")
        server_proc.kill()
        client_proc.kill()
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        server_proc.kill()
        client_proc.kill()
        return False


def test_simulator_only():
    """Prueba solo el simulador sin comunicación"""
    print("="*60)
    print("  TEST DEL SIMULADOR (SIN COMUNICACIÓN)")
    print("="*60)
    print()
    
    from robot_simulator import DifferentialRobotSimulator
    import random
    
    sim = DifferentialRobotSimulator()
    
    print("[1] Probando reset...")
    state = sim.reset()
    print(f"    Estado inicial: {state[:5]}... (truncado)")
    print(f"    Posición: ({state[0]:.2f}, {state[1]:.2f})")
    print(f"    Orientación: {state[2]:.2f} rad")
    
    print("\n[2] Probando pasos de simulación...")
    total_reward = 0
    for step in range(100):
        action = random.randint(0, 4)
        state, reward, done, info = sim.step(action)
        total_reward += reward
        
        if done:
            print(f"    Episodio terminado en paso {step+1}")
            print(f"    Objetivo alcanzado: {info['goal_reached']}")
            print(f"    Colisión: {info['collision']}")
            break
    
    print(f"    Recompensa total: {total_reward:.2f}")
    
    print("\n[3] Probando formato CSV...")
    state_csv = sim.get_state_string()
    print(f"    Estado CSV: {state_csv}")
    
    print("\n[4] Probando visualización ASCII...")
    print(sim.render_ascii())
    
    print("\n" + "-"*60)
    print("✓ Test del simulador completado!")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Tests del sistema')
    parser.add_argument('--test', choices=['simulator', 'communication', 'all'],
                        default='all', help='Tipo de test a ejecutar')
    args = parser.parse_args()
    
    results = []
    
    if args.test in ['simulator', 'all']:
        results.append(('Simulador', test_simulator_only()))
    
    if args.test in ['communication', 'all']:
        print("\n")
        results.append(('Comunicación TCP', test_basic_communication()))
    
    print("\n" + "="*60)
    print("  RESUMEN DE TESTS")
    print("="*60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
    print("="*60)


if __name__ == "__main__":
    main()
