import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import socket
import time

IMG_SIZE = 224

UMBRAL_CONFIANZA = 0.60
UMBRAL_SUPERPOSICION = 0.15

IP_RASPBERRY = "192.168.1.27"
PUERTO_UDP = 5000

TIEMPO_ENTRE_COMANDOS = 0.3
TIEMPO_SIN_SODA_MAX = 3.0


def construir_modelo_equilibrado():
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None
    )

    x = backbone.output

    x = layers.Conv2D(
        512,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    salida = layers.Conv2D(
        9,
        (1, 1),
        padding="same",
        activation="linear"
    )(x)

    return models.Model(inputs=backbone.input, outputs=salida)


modelo = construir_modelo_equilibrado()
modelo.load_weights("modelo_botellas_ssd_v2.h5")


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


udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

ultimo_envio = {
    "C": 0,
    "S": 0,
    "M": 0
}


def enviar_comando(comando):
    if comando not in ["C", "S", "M"]:
        return

    ahora = time.time()

    if ahora - ultimo_envio[comando] >= TIEMPO_ENTRE_COMANDOS:
        udp.sendto(comando.encode(), (IP_RASPBERRY, PUERTO_UDP))
        ultimo_envio[comando] = ahora


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    udp.close()
    exit()

NOMBRE_VENTANA = "Detector Laptop UDP"
cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)

ultimo_tiempo_soda = time.time()


while True:
    ret, frame = cap.read()

    if not ret:
        break

    alto_orig, ancho_orig, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_redimensionado = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    img_tensor = np.expand_dims(frame_redimensionado / 255.0, axis=0)

    prediccion = modelo.predict(img_tensor, verbose=0)[0]

    cajas_por_clase = {
        0: [],
        1: [],
        2: [],
        3: []
    }

    conf_por_clase = {
        0: [],
        1: [],
        2: [],
        3: []
    }

    for i in range(7):
        for j in range(7):

            confianza = 1.0 / (1.0 + np.exp(-prediccion[i, j, 0]))

            if confianza > UMBRAL_CONFIANZA:

                offset_x, offset_y, w, h = 1.0 / (
                    1.0 + np.exp(-prediccion[i, j, 1:5])
                )

                x_center = (j + offset_x) / 7.0
                y_center = (i + offset_y) / 7.0

                w = max(w, 0.10)
                h = max(h, 0.30)

                caja = [
                    y_center - (h / 2),
                    x_center - (w / 2),
                    y_center + (h / 2),
                    x_center + (w / 2)
                ]

                clase_id = np.argmax(prediccion[i, j, 5:9])

                cajas_por_clase[clase_id].append(caja)
                conf_por_clase[clase_id].append(confianza)

    detecto_coca = False
    detecto_salvietti = False
    detecto_alguna_soda = False

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

                detecto_alguna_soda = True

                if clase_id == 0:
                    detecto_coca = True

                if clase_id == 2:
                    detecto_salvietti = True

                cv2.rectangle(
                    frame,
                    (px_xmin, px_ymin),
                    (px_xmax, px_ymax),
                    color_caja,
                    3
                )

                cv2.putText(
                    frame,
                    f"{nombre_marca} {confianza_final * 100:.0f}%",
                    (px_xmin, max(px_ymin - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color_caja,
                    2
                )

    estado = "Sin Coca ni Salvietti"

    if detecto_alguna_soda:
        ultimo_tiempo_soda = time.time()

        if detecto_coca:
            enviar_comando("C")
            estado = "Coca detectada"

        if detecto_salvietti:
            enviar_comando("S")

            if detecto_coca:
                estado = "Coca y Salvietti"
            else:
                estado = "Salvietti detectada"

        if not detecto_coca and not detecto_salvietti:
            estado = "Otra soda detectada"

    else:
        tiempo_sin_soda = time.time() - ultimo_tiempo_soda
        estado = f"Sin soda: {tiempo_sin_soda:.1f} s"

        if tiempo_sin_soda > TIEMPO_SIN_SODA_MAX:
            enviar_comando("M")
            estado = "Sin soda > 3 s"

    cv2.putText(
        frame,
        estado,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.imshow(NOMBRE_VENTANA, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
udp.close()
