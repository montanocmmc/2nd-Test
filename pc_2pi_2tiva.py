import socket
import serial
import time


PUERTO_TIVA = "/dev/ttyACM0"
BAUD_TIVA = 9600

try:
    ser = serial.Serial(PUERTO_TIVA, BAUD_TIVA, timeout=1)
    time.sleep(2)
    print("UART con TIVA conectado.")

except Exception as e:
    print("No se pudo abrir el puerto serial de la TIVA.")
    print("Error:", e)
    exit()

HOST = "0.0.0.0"
PUERTO_UDP = 5000

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.bind((HOST, PUERTO_UDP))

print(f"Receptor UDP escuchando en puerto {PUERTO_UDP}...")
print("Esperando comandos..")


while True:
    try:
        data, addr = udp.recvfrom(1024)
        comando = data.decode(errors="ignore").strip()

        if comando in ["C", "S", "M"]:
            print(f"Recibido UDP desde {addr}: {comando}")
            ser.write((comando + "\n").encode())
            print(f"Enviado a TIVA por UART: {comando}")

        else:
            print(f"Comando ignorado desde {addr}: {comando}")

    except KeyboardInterrupt:
        print("Programa detenido.")
        break

    except Exception as e:
        print("Error:", e)


ser.close()
udp.close()

print("Puertos cerrados.")