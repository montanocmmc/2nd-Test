import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import socket
import time

#separacion de recursos graficos 
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

#iniciar gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Crecimiento dinámico de VRAM habilitado.")
    except RuntimeError as e:
        print(f"Error al configurar la GPU: {e}")

IMG_SIZE = 224
UMBRAL_CONFIANZA = 0.4
UMBRAL_SUPERPOSICION = 0.15 

IP_RASPBERRY = "192.168.1.27"  
PUERTO_UDP = 5000

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.setblocking(False)

sistema_activo = True 
conteo_iniciado = False
DURACION_MUESTREO = 60  
tiempo_acumulado = 0.0
ultimo_tick = 0.0

def construir_modelo_equilibrado():
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=None
    )
    x = backbone.output 
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', 
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x) 
    salida = layers.Conv2D(9, (1, 1), padding='same', activation='linear')(x)
    return models.Model(inputs=backbone.input, outputs=salida)

print("Cargando modelo en la Computadora...")
modelo = construir_modelo_equilibrado()
modelo.load_weights('modelo_botellas_ssd_v2.h5')
print("¡Modelo cargado exitosamente!")

NOMBRES_CLASES = {0: "Coca", 1: "Fanta", 2: "Salvietti", 3: "Pepsi"}
COLORES = {0: (0, 0, 255), 1: (0, 165, 255), 2: (0, 255, 0), 3: (255, 0, 0)}

inventario_real = {0: 0, 1: 0, 2: 0, 3: 0}
TAMAÑO_BUFFER = 3  
buffers = {0: [], 1: [], 2: [], 3: []}
estado_estable = {0: False, 1: False, 2: False, 3: False}

cap = cv2.VideoCapture(0)
NOMBRE_VENTANA = 'Detector Computadora - UDP Bidireccional'
cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)

print("Cámara iniciada. Presiona cualquier tecla en la ventana de video para arrancar el tiempo.")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        alto_orig, ancho_orig, _ = frame.shape
        ahora = time.time()
        try:
            data, addr_in = udp.recvfrom(1024)
            comando_entrante = data.decode().strip()
            if comando_entrante == "PAUSA":
                sistema_activo = False
                print("\n[RED] ¡Sistema pausado desde el botón de la Raspberry Pi!")
            elif comando_entrante == "REANUDAR":
                sistema_activo = True
                print("\n[RED] Sistema reanudado desde la Raspberry Pi.")
        except BlockingIOError:
            pass 

        if not conteo_iniciado:
            cv2.putText(frame, "PRESIONA CUALQUIER TECLA PARA INICIAR", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Acumulamos tiempo SOLO si el sistema no está pausado
            if sistema_activo:
                tiempo_acumulado += (ahora - ultimo_tick)
            ultimo_tick = ahora  # Siempre actualizamos el tick para que no de "saltos" de tiempo

            tiempo_restante = max(0.0, DURACION_MUESTREO - tiempo_acumulado)
            
            # Textos en pantalla dependientes del estado
            if sistema_activo:
                cv2.putText(frame, f"TIEMPO RESTANTE: {int(tiempo_restante)}s", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "SISTEMA PAUSADO DESDE LA PI", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            # Dibujar el inventario en tiempo real en la pantalla
            y_pos = 80
            for id_clase, cantidad in inventario_real.items():
                texto_inv = f"{NOMBRES_CLASES[id_clase]}: {cantidad}"
                cv2.putText(frame, texto_inv, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORES[id_clase], 2)
                y_pos += 30

            if tiempo_acumulado >= DURACION_MUESTREO:
                print("Tiempo concluido. Notificando a la Pi...")
                udp.sendto("END".encode(), (IP_RASPBERRY, PUERTO_UDP))
                break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_redimensionado = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
        img_tensor = np.expand_dims(frame_redimensionado / 255.0, axis=0)
        prediccion = modelo.predict(img_tensor, verbose=0)[0]

        cajas_por_clase = {0: [], 1: [], 2: [], 3: []}
        conf_por_clase = {0: [], 1: [], 2: [], 3: []}

        for i in range(7): 
            for j in range(7):
                confianza = 1.0 / (1.0 + np.exp(-prediccion[i, j, 0]))
                if confianza > UMBRAL_CONFIANZA:
                    offset_x, offset_y, w, h = 1.0 / (1.0 + np.exp(-prediccion[i, j, 1:5]))
                    x_center, y_center = (j + offset_x) / 7.0, (i + offset_y) / 7.0
                    w, h = max(w, 0.10), max(h, 0.30)
                    caja = [y_center - (h / 2), x_center - (w / 2), y_center + (h / 2), x_center + (w / 2)]
                    clase_id = np.argmax(prediccion[i, j, 5:9])
                    cajas_por_clase[clase_id].append(caja)
                    conf_por_clase[clase_id].append(confianza)

        detecciones_este_frame = {0: False, 1: False, 2: False, 3: False}

        for clase_id in range(4):
            if len(cajas_por_clase[clase_id]) > 0:
                boxes_np = np.array(cajas_por_clase[clase_id], dtype=np.float32)
                scores_np = np.array(conf_por_clase[clase_id], dtype=np.float32)
                indices_aprobados = tf.image.non_max_suppression(
                    boxes_np, scores_np, max_output_size=5, iou_threshold=UMBRAL_SUPERPOSICION
                )
                if len(indices_aprobados) > 0:
                    detecciones_este_frame[clase_id] = True

                color_caja = COLORES[clase_id]
                for idx in indices_aprobados:
                    y_min, x_min, y_max, x_max = boxes_np[int(idx)]
                    px_xmin, px_ymin = int(x_min * ancho_orig), int(y_min * alto_orig)
                    px_xmax, px_ymax = int(x_max * ancho_orig), int(y_max * alto_orig)
                    cv2.rectangle(frame, (px_xmin, px_ymin), (px_xmax, px_ymax), color_caja, 3)
                    cv2.putText(frame, f"{NOMBRES_CLASES[clase_id]} {scores_np[int(idx)]*100:.0f}%", 
                                (px_xmin, px_ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_caja, 2)

        # Filtro Anti-Rebote (Se bloquea si el sistema está en PAUSA)
        if conteo_iniciado and sistema_activo:
            for clase_id in range(4):
                buffers[clase_id].append(detecciones_este_frame[clase_id])
                if len(buffers[clase_id]) > TAMAÑO_BUFFER:
                    buffers[clase_id].pop(0)

                if len(buffers[clase_id]) == TAMAÑO_BUFFER:
                    if all(buffers[clase_id]) and not estado_estable[clase_id]:
                        estado_estable[clase_id] = True
                        inventario_real[clase_id] += 1
                        comando = NOMBRES_CLASES[clase_id].upper()
                        udp.sendto(comando.encode(), (IP_RASPBERRY, PUERTO_UDP))
                        print(f"[{int(tiempo_acumulado)}s] {comando} contada y notificada.")

                    elif not any(buffers[clase_id]) and estado_estable[clase_id]:
                        estado_estable[clase_id] = False

        cv2.imshow(NOMBRE_VENTANA, frame)
        tecla = cv2.waitKey(1) & 0xFF
        
        if not conteo_iniciado and tecla != 255 and tecla != ord('q'): 
            conteo_iniciado = True
            ultimo_tick = time.time()
            udp.sendto("START".encode(), (IP_RASPBERRY, PUERTO_UDP))
            print("\n=== INICIANDO CONTEO DE 1 MINUTO ===")
            
        elif tecla == ord('q') or tecla == ord('Q'): 
            udp.sendto("END".encode(), (IP_RASPBERRY, PUERTO_UDP))
            break

except KeyboardInterrupt:
    print("\n[AVISO] Se detuvo el programa desde la terminal. Generando gráfica de todos modos...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    udp.close()

    # ============================================================
    # GENERACIÓN Y DESPLIEGUE DE LA GRÁFICA GARANTIZADA
    # ============================================================
    print("\nProcesando gráfica final...")
    resultados_finales = {NOMBRES_CLASES[id_clase]: conteo for id_clase, conteo in inventario_real.items()}
    top_3 = sorted(resultados_finales.items(), key=lambda x: x[1], reverse=True)[:3]
    marcas = [item[0] for item in top_3]
    valores = [item[1] for item in top_3]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(marcas, valores, color=['#FF5733', '#33FF57', '#3357FF'])
    plt.title('Top 3 Botellas Contadas (Procesadas en Computadora)')
    plt.xlabel('Marca')
    plt.ylabel('Unidades Detectadas')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(max(valores)*0.01, 0.1)), int(yval), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("grafica_botellas_PC.png")  
    print("Gráfica guardada como 'grafica_botellas_PC.png'. Abriendo ventana...")
    plt.show()
