import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)
hand = mp.solutions.hands                           #fornece o mapeamento das mãos
Hand = hand.Hands(max_num_hands=2)                  #definimos o número máximo de mãos a serem detectadas
mpDraw = mp.solutions.drawing_utils

while True:
    check,img = video.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = Hand.process(imgRGB)                  #processará a imagem em RGB
    handsPoints = results.multi_hand_landmarks      #retorna pontos na imagem
    h,w,_ = img.shape                               #extração das dimensões da imagem

    totalDedos = 0 # Variável para armazenar a soma dos dedos das duas mãos

    if handsPoints: #necessário, pois caso detectada a mão ele irá imprimir as posições
        for points in handsPoints:
            #print(points)
            mpDraw.draw_landmarks(img,points,hand.HAND_CONNECTIONS)  #fará o desenho das linhas na mão
            pontos = [(int(cord.x * w), int(cord.y * h)) for cord in points.landmark]

            for id,cord in enumerate(points.landmark):               #irá enumerar cada ponto da mão
                cx,cy = int(cord.x*w), int(cord.y*h)
                cv2.putText(img,str(id),(cx,cy+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
                pontos.append((cx,cy))

            if len(pontos) >= 18:  # Garante que temos os pontos necessários
                dedoM = pontos[12]  # Ponto 0 (pulso)
                baseIndicador = pontos[5]  # Ponto 5 (base do dedo indicador)
                baseMinimo = pontos[17]  # Ponto 17 (base do dedo mínimo)

                # Identificar se é mão direita ou esquerda
                if baseIndicador[0] > baseMinimo[0]:
                    mao = "Direita"
                else:
                    mao = "Esquerda"

                # Exibir o rótulo da mão próximo ao dedo médio
                cv2.putText(img, mao, (dedoM[0] - 20, dedoM[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                dedos=[8,12,16,20] #conterá os pontos extremos superiores dos dedos
                contador=0

                for dedo in dedos:
                    if pontos[dedo][1] < pontos[dedo - 2][1]:  # O topo do dedo está acima da base
                        contador += 1

                # Contar o polegar (diferente para cada mão)
                if mao == "Direita":
                    if pontos[4][0] > pontos[2][0]:  # Polegar à direita do dedo indicador
                        contador += 1
                else:  # Mão esquerda
                    if pontos[4][0] < pontos[2][0]:  # Polegar à esquerda do dedo indicador
                        contador += 1

                # Somar a contagem de dedos das mãos detectadas
                totalDedos += contador

                if totalDedos < 10:
                    cv2.rectangle(img,(80,10),(200,110),(0,0,0),-1)
                    cv2.putText(img,str(totalDedos),(100,100),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),5)
                else:
                    cv2.rectangle(img, (80, 10), (260, 110), (0, 0, 0), -1)
                    cv2.putText(img, str(totalDedos), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)

    cv2.imshow("Imagem",img)
    cv2.waitKey(1)
