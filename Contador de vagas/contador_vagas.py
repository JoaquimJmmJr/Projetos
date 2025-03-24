import cv2
import pickle
import numpy as np

vagas = []
with open('vagas.pkl','rb') as arquivo: #vai ler o arquivo com 'rb'
    vagas = pickle.load(arquivo)        #vai carregar o arquivo

video = cv2.VideoCapture('video.mp4')

while True:
    check,img = video.read()
    imgCinza = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgTh = cv2.adaptiveThreshold(imgCinza,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,16)
    #pixels com mais intensidade serão transformados em branco, e com menos em preto
    #o 25 e 16 variam de projeto para projeto
    imgMedian = cv2.medianBlur(imgTh,5) #técnica para limpar a imagem e tirar os ruídos
    kernel = np.ones((3,3),np.int8)
    imgDil = cv2.dilate(imgMedian,kernel)

    vagasAbertas = 0
    for x,y,w,h in vagas:
        vaga = imgDil[y:y+h,x:x+w]
        count = cv2.countNonZero(vaga) #essa função vai calcular a quantidade de pixels brancos que não são zero
        cv2.putText(img,str(count),(x,y+h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        if count < 910:               #para uma contagem menor do que 1000 pixels brancos = vaga vazia
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            vagasAbertas += 1
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.rectangle(img,(90,0),(415,60),(0,255,0),-1)
        cv2.putText(img,f'LIVRE: {vagasAbertas}/69',(95,45),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),5)

    cv2.imshow('video Threshold', imgTh)
    #cv2.imshow('video Median Blur', imgMedian)
    cv2.imshow('video',img)
    cv2.waitKey(20)
