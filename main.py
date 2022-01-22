# Imports
from wsgiref.validate import validator
import numpy as np
import cv2
import math
import time
from button import Button

# Open Camera
capture = cv2.VideoCapture(0)
buttonList=[]
buttonListValues = [['7','8','9','*'],
                    ['4','5','6','-'],
                    ['1','2','3','+'],
                    ['0','/','.','=']
                    ]

pTime=0
for x in range(4):
    for y in range(4):
        xpos=x*75 
        ypos=y*75 
        buttonList.append(Button((xpos,ypos),75,75,buttonListValues[y][x]))

button1=Button((100,100),50,50,'7')
auxStart=[]
auxEnd=0
auxFar=0
k=0
result=' '
flag=False
select=False
canSelect=True
hasAngle=False
myEquation=' '
while capture.isOpened():

    # Capture frames from the camera
    ret, frame = capture.read()
    frame=cv2.flip(frame,1)
    cv2.circle(frame,(260,260),5,(0,0,0))
    cv2.rectangle(frame,(100,25),(400,100),(0,0,0),2)
   
    # Get hand data from the rectangle sub window
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 0)
    crop_image = frame[100:400, 100:400]
    for button in buttonList:
        button.draw(crop_image)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # Change color-space from BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # Kernel for morphological transformation
    kernel = np.ones((5, 5))

    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    # Show threshold image
    cv2.imshow("Thresholded", thresh)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # Find contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Create bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Find convex hull
        hull = cv2.convexHull(contour)

        # Draw contour
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Find convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
        # tips) for all defects
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = int((math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14)
            cv2.line(crop_image, start, end, [0, 255, 0], 2)
            # print(start[0])
            print(' angle:',angle)
            if angle<30:
               hasAngle=True
               auxStart=start
               break
                
            else:
                hasAngle =False
        # k=0 mão aberta(>30º)
        # k=1 mão a fechar antes de ter o valor (<30º)
        # k=2 mão a fechar depois de ter o valor  (<30º)
        # k=3 mão fechada (>30º)
        # k=4 mão a abrir (<30º)
        if hasAngle ==True:
            if k==0:
                k=1
            elif k==3:
                k=4
                flag=True

        if hasAngle ==False:
            if k==2:
                k=3
            elif k==4:
                k=0
               
   
        if k==1 :
            cv2.circle(crop_image, start, 5, [255, 0, 0], -1)
            cv2.circle(crop_image, end, 5, [255, 0, 0], -1)
            cv2.circle(crop_image, far, 5, [0, 0, 255], -1)
            for button in buttonList:
                myValue=button.getValue(auxStart[0],auxStart[1],angle)
                # if k==1:
                if myValue != 'x' :
                    k=2
                    select=True
                    # canSelect=False
                    if(myValue=='='):
                        myEquation = str(eval(myEquation))
                    else:
                        myEquation += myValue
                
            flag=True
        # print(int(angle))

       

        # if exist==False :
            
        
    except:
        pass
    
    # if result[len(result)-1]=='=':
    #    cv2.putText(frame, str(eval(result[:-1])), (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)

    # else:
    cv2.putText(frame, str(myEquation), (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)
    print(result)
    # Show required images
    cv2.imshow("Gesture", frame)
    all_image = np.hstack((drawing, crop_image))
    cv2.imshow('Contours', all_image)

    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()