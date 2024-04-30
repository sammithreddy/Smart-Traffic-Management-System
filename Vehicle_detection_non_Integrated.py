import cv2
import numpy as np
from time import sleep

right_min=80 
left_min=80 

offset=6   

pos=550 

delay= 60

detec = []
count= 0

	
def count_of_vehicles(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('C:/Users/DELL/Desktop/Coding/Projects/Detecting no.of vehicles/video.mp4')
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret , frame1 = cap.read()
    tempo = float(1/delay)
    sleep(tempo) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    simulator = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    simulator = cv2.morphologyEx (simulator, cv2. MORPH_CLOSE , kernel)
    contorno,h=cv2.findContours(simulator,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, pos), (1200, pos), (255,127,0), 3) 
    for(i,c) in enumerate(contorno):
        (x,y,w,h) = cv2.boundingRect(c)
        validator = (w >= right_min) and (h >= left_min)
        if not validator:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        centro = count_of_vehicles(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0,255), -1)

        for (x,y) in detec:
            if y<(pos+offset) and y>(pos-offset):
                count+=1
                cv2.line(frame1, (25, pos), (1200, pos), (0,127,255), 3)  
                detec.remove((x,y))
                print("car is detected : "+str(count))        
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detectar",simulator)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
