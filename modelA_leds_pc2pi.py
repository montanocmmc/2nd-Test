import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import socket
import subprocess
import time



IMG_SIZE = 224
UMBRAL_CONFIANZA = 0.60
UMBRAL_SUPERPOSICION = 0.15
TIEMPO_ENTRE_COMANDOS = 1.0

#Modelo

def construir_modelo_equilibrado():
    backbone = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,weights=None)
    x = backbone.output
    x = layers.Conv2D(512,(3, 3),padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    salida = layers.Conv2D(9,(1, 1),padding="same",activation="linear")(x)

    return models.Model(inputs=backbone.input, outputs=salida)


print("Cargando modelo...")
modelo = construir_modelo_equilibrado()
modelo.load_weights("modelo_botellas_ssd_v2.h5")
print("Modelo cargado exitosamente.")


NOMBRES_CLASES = {
    0: "Coca",
    1: "Fanta",
    2: "Salvietti",
    3: "Pepsi"
}

COLORES = {
    0: (0, 0, 255),
    1: (0, 165, 255),
    2: (0, 255, 0),
    3: (255, 0, 0)
}


IP_RASPBERRY = "192.168.1.27"
USUARIO_RASPBERRY = "cactus"

proceso_ssh = subprocess.Popen(
    [
        "ssh",
        f"{USUARIO_RASPBERRY}@{IP_RASPBERRY}",
        "python3 ~/parcial-2/part3ssh.py"
    ],
    stdin=subprocess.PIPE,
    text=True
)

ultimo_comando_enviado = None
def enviar_comando(comando):
    if comando not in ["C", "S"]:
        return

    proceso_ssh.stdin.write(comando + "\n")
    proceso_ssh.stdin.flush()

    print(f"Comando enviado por SSH: {comando}")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Cámara iniciada. Presiona 'q' para salir.")

NOMBRE_VENTANA = "Detector Laptop TCP"
cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)



while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer la cámara.")
        break

    alto_orig, ancho_orig, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_redimensionado = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    img_tensor = np.expand_dims(frame_redimensionado / 255.0, axis=0)

    prediccion = modelo.predict(img_tensor, verbose=0)[0]

    cajas_por_clase = {0:[], 1:[], 2:[], 3: []}
    conf_por_clase = {0:[], 1:[], 2:[], 3:[]}

    for i in range(7):
        for j in range(7):
            confianza = 1.0 / (1.0 + np.exp(-prediccion[i, j, 0]))
            if confianza > UMBRAL_CONFIANZA:
                offset_x, offset_y, w, h = 1.0 / (1.0 + np.exp(-prediccion[i, j, 1:5]))
                x_center = (j + offset_x) / 7.0
                y_center = (i + offset_y) / 7.0
                w = max(w, 0.10)
                h = max(h, 0.30)
                caja = [y_center - (h / 2), x_center - (w / 2), y_center + (h / 2), x_center + (w / 2)]
                clase_id = np.argmax(prediccion[i, j, 5:9])
                cajas_por_clase[clase_id].append(caja)
                conf_por_clase[clase_id].append(confianza)

    detecto_coca = False
    detecto_salvietti = False
    for clase_id in range(4):
        if len(cajas_por_clase[clase_id]) > 0:
            boxes_np = np.array(cajas_por_clase[clase_id], dtype=np.float32)
            scores_np = np.array(conf_por_clase[clase_id], dtype=np.float32)
            indices_aprobados = tf.image.non_max_suppression(
                boxes_np,
                scores_np,
                max_output_size=5,
                iou_threshold=UMBRAL_SUPERPOSICION
            )

            color_caja = COLORES[clase_id]
            nombre_marca = NOMBRES_CLASES[clase_id]

            for idx in indices_aprobados:
                idx = int(idx)

                y_min, x_min, y_max, x_max = boxes_np[idx]
                confianza_final = scores_np[idx]

                x_min = max(0, min(1, x_min))
                y_min = max(0, min(1, y_min))
                x_max = max(0, min(1, x_max))
                y_max = max(0, min(1, y_max))

                px_xmin = int(x_min * ancho_orig)
                px_ymin = int(y_min * alto_orig)
                px_xmax = int(x_max * ancho_orig)
                px_ymax = int(y_max * alto_orig)

                if clase_id == 0:
                    detecto_coca = True

                if clase_id == 2:
                    detecto_salvietti = True

                cv2.rectangle(frame,(px_xmin, px_ymin),(px_xmax, px_ymax),color_caja,3)
                cv2.putText(frame,f"{nombre_marca} {confianza_final * 100:.0f}%",(px_xmin, max(px_ymin - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,color_caja,2)

    estado = "Sin Coca ni Salvietti"
    if detecto_coca:
        enviar_comando("C")
        estado = "Coca detectada"
    if detecto_salvietti:
        enviar_comando("S")

        if detecto_coca:
            estado = "Coca y Salvietti detectadas"
        else:
            estado = "Salvietti detectada"

    cv2.putText(frame,estado,(30, 40),cv2.FONT_HERSHEY_SIMPLEX,0.75,
        (255, 255, 255),2)

    cv2.imshow(NOMBRE_VENTANA, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
print("Programa finalizado.")