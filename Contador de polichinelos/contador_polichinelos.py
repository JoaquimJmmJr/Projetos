import cv2
import mediapipe as mp
import math
from fontTools.misc.arrayTools import pointsInRect

video = cv2.VideoCapture('polichinelos.mp4') #abre o vídeo
pose = mp.solutions.pose
Pose = pose.Pose(min_tracking_confidence=0.5,min_detection_confidence=0.5) #variável responsável pela detecção
draw = mp.solutions.drawing_utils #irá detectar os pontos dentro da imagem
contador = 0 #iniciando o contador para os polichinelos
check = True

while True:
    success,img = video.read()
    videoRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #converte o vídeo para RGB
    results = Pose.process(videoRGB) #processará a imagem e devolverá os pontos do corpo de dentro do vídeo
    points = results.pose_landmarks  #extraindo os pontos de dentro da imagem e retornando as coordenadas deles
    draw.draw_landmarks(img,points,pose.POSE_CONNECTIONS)
    h,w,_ = img.shape #extração das dimensões da imagem


    #IDENTIFICAÇÃO DOS PONTOS DE REFERÊNCIA PARA COMPLETAR UM POLICHINELO
    # #conversão dos pontos em pixels (x*w) e (y*h)
    if points:
        peDX = int(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].x*w)
        peDY = int(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].y*h)

        peEX = int(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].x*w)
        peEY = int(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].y*h)

        maoDX = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].x*w)
        maoDY = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].y*h)

        maoEX = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].x*w)
        maoEY = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].y*h)

        distMAOS = math.hypot(maoDX-maoEX,maoDY-maoEY)
        distPES = math.hypot(peDX-peEX,peDY-peEY)

        print(f'maos: {distMAOS} ,pes:{distPES}')
        if check == True and distMAOS <= 105 and distPES >= 170:
            contador +=1
            check = False
            #o check garante que o polichinelo só será contabilizado 1 vez
        else:
            check = True

        texto = f'Quant.: {contador}'
        cv2.rectangle(img,(20,240),(350,120),(255,0,0),-1)
        cv2.putText(img,texto,(40,200),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),5)

    cv2.imshow('Resultado',img)
    cv2.waitKey(5)
