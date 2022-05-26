import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # classifier is a detector for the faces

# To capture video from webcam.
webcam = cv2.VideoCapture(0) # 0 is the webcam by default

while True:
    successful_frame_read, frame = webcam.read()                                     
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    face_coordinates = trained_face_data.detectMultiScale(gray_img) # AI detects faces
    # Shows the image 
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    cv2.imshow('Face Detection', frame) 
    cv2.waitKey(1) # auto refreshs the image of the webcan every 1 ms
