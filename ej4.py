import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import time
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error de GPU: {e}")

IMG_SIZE = 224

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

print("Construyendo arquitectura y cargando pesos...")
modelo = construir_modelo_equilibrado()
modelo.load_weights('modelo_botellas_ssd_v2.h5')
print("¡Modelo cargado exitosamente!")

NOMBRES_CLASES = {0: "Coca", 1: "Fanta", 2: "Salvietti", 3: "Pepsi"}
COLORES = {0: (0, 0, 255), 1: (0, 165, 255), 2: (0, 255, 0), 3: (255, 0, 0)}
UMBRAL_CONFIANZA = 0.4
UMBRAL_SUPERPOSICION = 0.15 

TAMAÑO_BUFFER = 3  
buffers = {0: [], 1: [], 2: [], 3: []}
estado_estable = {0: False, 1: False, 2: False, 3: False}
inventario_real = {0: 0, 1: 0, 2: 0, 3: 0}

conteo_iniciado = False
tiempo_inicio = 0
DURACION_MUESTREO = 60  

cap = cv2.VideoCapture(0)
NOMBRE_VENTANA = 'Detector Anti-Rebote'
cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)

print("Iniciando transmisión. Presiona cualquier tecla para empezar el conteo de 60s.")

while True:
    ret, frame = cap.read()
    if not ret: break

    alto_orig, ancho_orig, _ = frame.shape
    
    if not conteo_iniciado:
        cv2.putText(frame, "PRESIONA CUALQUIER TECLA PARA INICIAR (60s)", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        tiempo_transcurrido = time.time() - tiempo_inicio
        tiempo_restante = DURACION_MUESTREO - tiempo_transcurrido
        cv2.putText(frame, f"TIEMPO RESTANTE: {int(tiempo_restante)}s", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        y_pos = 80
        for id_clase, cantidad in inventario_real.items():
            texto_inv = f"{NOMBRES_CLASES[id_clase]}: {cantidad}"
            cv2.putText(frame, texto_inv, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORES[id_clase], 2)
            y_pos += 30

        if tiempo_transcurrido >= DURACION_MUESTREO:
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

    if conteo_iniciado:
        for clase_id in range(4):
            buffers[clase_id].append(detecciones_este_frame[clase_id])
            if len(buffers[clase_id]) > TAMAÑO_BUFFER:
                buffers[clase_id].pop(0)

            if len(buffers[clase_id]) == TAMAÑO_BUFFER:
                if all(buffers[clase_id]) and not estado_estable[clase_id]:
                    estado_estable[clase_id] = True
                    inventario_real[clase_id] += 1
                    print(f"[{int(tiempo_transcurrido)}s] ¡Nueva {NOMBRES_CLASES[clase_id]} estabilizada y contada!")

                elif not any(buffers[clase_id]) and estado_estable[clase_id]:
                    estado_estable[clase_id] = False

    cv2.imshow(NOMBRE_VENTANA, frame)

    tecla = cv2.waitKey(1) & 0xFF
    if not conteo_iniciado and tecla != 255: 
        conteo_iniciado = True
        tiempo_inicio = time.time()
        print("\n=== INICIANDO CONTEO DE 60 SEGUNDOS ===")
    elif tecla == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
print("\n=== TIEMPO FINALIZADO ===")

resultados_finales = {NOMBRES_CLASES[id_clase]: conteo for id_clase, conteo in inventario_real.items()}

nombre_archivo = "resultados_fisicos_botellas.txt"
with open(nombre_archivo, "w") as archivo:
    archivo.write("--- INVENTARIO FÍSICO DE BOTELLAS (60 SEGUNDOS) ---\n")
    archivo.write("Método: Detección SSD + Filtro Anti-Rebote (Debounce)\n\n")
    for marca, total in resultados_finales.items():
        archivo.write(f"Marca: {marca} | Unidades físicas contadas: {total}\n")
print(f"Archivo guardado como: {nombre_archivo}")

top_3 = sorted(resultados_finales.items(), key=lambda x: x[1], reverse=True)[:3]
marcas = [item[0] for item in top_3]
valores = [item[1] for item in top_3]

plt.figure(figsize=(8, 6))
bars = plt.bar(marcas, valores, color=['#FF5733', '#33FF57', '#3357FF'])
plt.title('Top 3 Botellas Físicas Contadas (1 Minuto)')
plt.xlabel('Marca')
plt.ylabel('Unidades Físicas')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (max(max(valores)*0.01, 0.1)), int(yval), ha='center', va='bottom')

plt.tight_layout()
plt.show()