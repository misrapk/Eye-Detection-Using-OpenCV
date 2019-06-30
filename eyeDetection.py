import cv2
face_cascade = cv2.CascadeClassifier('E:\piyush\coding\Training Ducat\documents of ml\haarcascade_frontalface_default.xml')
eye_cascade= cv2.CascadeClassifier('E:\piyush\coding\Training Ducat\documents of ml\haarcascade_eye.xml')


cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    print(ret)
    
    #convert into gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    #for face 
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
        
        #for specifyng certain area of face
        roi_gray = gray [y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        #detect eyes 
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew , ey+eh), (0,255,255), 2)
        
    cv2.imshow('img', img)
    key = cv2.waitKey(30) & 0xFF
    if key ==27: 
        break
        
cap.release()     #it will release the camera
cv2.destroyAllWindows()