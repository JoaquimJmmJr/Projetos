import cv2
import mediapipe as mp  # Certifique-se de que está instalado
import math
import time
import serial           # Biblioteca para fazer a comunicação com o arduino

#Funciona nas versões:
#python: 3.10
#opencv-python: 4.11.0.86
#mediapipe: 0.10.21

# Configuração da câmera e do Mediapipe
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# Inicialização de variáveis
status = ""  # Inicializa status antes do loop
inicio = None  # Variável para marcar o início do tempo
situacao = "A"

# Configuração da comunicação com Arduino (ajuste a porta)
arduino = serial.Serial('COM9', 9600)  # Substitua pela porta correta
time.sleep(2)  # Aguarda a conexão estabilizar

while True:
    check, img = video.read()
    img = cv2.resize(img, (800, 720))
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    h, w, _ = img.shape

    if results and results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            # Captura pontos dos olhos
            di1x, di1y = int(face.landmark[159].x * w), int(face.landmark[159].y * h)
            di2x, di2y = int(face.landmark[145].x * w), int(face.landmark[145].y * h)
            ei1x, ei1y = int(face.landmark[386].x * w), int(face.landmark[386].y * h)
            ei2x, ei2y = int(face.landmark[374].x * w), int(face.landmark[374].y * h)

            # Desenha pontos nos olhos
            for x, y in [(di1x, di1y), (di2x, di2y), (ei1x, ei1y), (ei2x, ei2y)]:
                cv2.circle(img, (x, y), 1, (255, 0, 0), 2)

            # Calcula a distância dos olhos
            distDi = math.hypot(di1x - di2x, di1y - di2y)
            distEs = math.hypot(ei1x - ei2x, ei1y - ei2y)

            if distEs <= 10 and distDi <= 10:
                # Olhos fechados
                cv2.rectangle(img, (100, 30), (390, 80), (0, 0, 255), -1)
                cv2.putText(img, "EYES: CLOSED", (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                situacao = "F"

                if situacao != status:
                    inicio = time.time()  # Marca o início do tempo se ainda não foi marcado

                tempo = int(time.time() - inicio) if inicio else 0  # Calcula o tempo desde que os olhos fecharam
            else:
                # Olhos abertos
                cv2.rectangle(img, (100, 30), (390, 80), (0, 255, 0), -1)
                cv2.putText(img, "EYES: OPEN", (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                situacao = "A"
                inicio = None  # Reseta o tempo quando os olhos são abertos
                tempo = 0

            status = situacao  # Atualiza status

            # Lógica do buzzer
            if situacao == "F" and tempo >= 2:
                cv2.rectangle(img, (300, 150), (850, 220), (0, 0, 255), -1)
                cv2.putText(img, f'SLEEPING {tempo}S', (310, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 5)
                arduino.write(b'ON\n')  # Liga o buzzer
                print("Buzzer: ON")
                print("tempo:",tempo,"segundos")
            else:
                arduino.write(b'OFF\n')  # Desliga o buzzer
                print("Buzzer: OFF")

    cv2.imshow('Imagem', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Encerra conexões ao sair do loop
arduino.close()
video.release()
cv2.destroyAllWindows()
