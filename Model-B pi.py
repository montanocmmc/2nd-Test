import socket
import time
import RPi.GPIO as GPIO

inventario_real = {0: 0, 1: 0, 2: 0, 3: 0}
NOMBRES_CLASES = {0: "Coca", 1: "Fanta", 2: "Salvietti", 3: "Pepsi"}

sistema_activo = True
conteo_iniciado = False
PIN_BOTON_EMERGENCIA = 23
ultimo_tick_pi = 0.0
tiempo_acumulado_escaneo = 0.0

direccion_pc = None 

ultimo_estado_boton = GPIO.HIGH
ultimo_tiempo_debounce = 0.0
TIEMPO_DEBOUNCE = 0.3  


def exportar_reporte_txt(nombre_archivo, titulo_encabezado):
    with open(nombre_archivo, "w", encoding="utf-8") as archivo:
        archivo.write(f"--- {titulo_encabezado} ---\n")
        archivo.write(f"Tiempo: {tiempo_acumulado_escaneo:.2f} segundos\n\n")
        for id_clase, total in inventario_real.items():
            archivo.write(f"Marca: {NOMBRES_CLASES[id_clase]} | Unidades contadas: {total}\n")
    print(f"\n[DISCO LOCAL PI] Archivo guardado {nombre_archivo}")


GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_BOTON_EMERGENCIA, GPIO.IN, pull_up_down=GPIO.PUD_UP)


HOST = "0.0.0.0"
PUERTO_UDP = 5000

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.bind((HOST, PUERTO_UDP))
# Definimos un timeout de 100ms para que recvfrom permita leer el botón
udp.settimeout(0.1) 

print(f"Puerto {PUERTO_UDP}...")


while True:
    try:
        ahora = time.time()
        
        if conteo_iniciado and sistema_activo:
            tiempo_acumulado_escaneo += (ahora - ultimo_tick_pi)
        ultimo_tick_pi = ahora

        estado_actual_boton = GPIO.input(PIN_BOTON_EMERGENCIA)
        
        if estado_actual_boton == GPIO.LOW and ultimo_estado_boton == GPIO.HIGH:
            if (ahora - ultimo_tiempo_debounce) > TIEMPO_DEBOUNCE:
                ultimo_tiempo_debounce = ahora
                sistema_activo = not sistema_activo 
                
                if direccion_pc is not None:
                    if not sistema_activo:
                        print("\n[BOTÓN] Sistema Pausado")
                        udp.sendto("PAUSA".encode(), direccion_pc)
                    else:
                        udp.sendto("REANUDAR".encode(), direccion_pc)
                else:
                    print("\n[AVISO] Botón presionado")
                        
        ultimo_estado_boton = estado_actual_boton

        try:
            data, addr = udp.recvfrom(1024)
            comando = data.decode(errors="ignore").strip()
            
            if direccion_pc is None:
                direccion_pc = addr
                
        except socket.timeout:
            continue 

        if comando == "START":
            conteo_iniciado = True
            print("\n[REDEVENT] Escaneo iniciado")

        elif comando == "END":
            exportar_reporte_txt("resultados_fisicos_botellas.txt", "INVENTARIO FÍSICO FINAL DE BOTELLAS")
            break

        elif sistema_activo and conteo_iniciado:
            if comando == "COCA":
                inventario_real[0] += 1
                print(f"[CONTEO] Coca sumada -> Total: {inventario_real[0]}")
            elif comando == "FANTA":
                inventario_real[1] += 1
                print(f"[CONTEO] Fanta sumada -> Total: {inventario_real[1]}")
            elif comando == "SALVIETTI":
                inventario_real[2] += 1
                print(f"[CONTEO] Salvietti sumada -> Total: {inventario_real[2]}")
            elif comando == "PEPSI":
                inventario_real[3] += 1
                print(f"[CONTEO] Pepsi sumada -> Total: {inventario_real[3]}")
                
        elif not sistema_activo:
            print(f"[IGNORADO] Comando '{comando}' bloqueado por estado de Pausa.")

    except KeyboardInterrupt:
        break
    except Exception as e:
        print("Error", e)


udp.close()
GPIO.cleanup()