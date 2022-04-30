import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # classifier is a detector for the faces

# To capture video from webcam.
webcam = cv2.VideoCapture(0) # 0 is the webcam by default

# Iterate forever over frames
while True:
    #### Read the current frame
    successful_frame_read, frame = webcam.read() # successful_frame_read is if the frame was succesfull read (always True)
                                                 # the frame is the image of the webcam
    # must convert to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # cvt = convert / 1st arg = what image, 2st arg = what type of convertion

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(gray_img) # detectMultiScale = Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.

    # Draw rectangles around the faces
    # cv2.rectangle(img, (x, y), (x+width, y+hight), (0, 255, 0), 2)
    
    # Shows the image 
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # print face coordinates to find out what to put in the first 2 parentesis
                                                         # 0, 255, 0 is teh color RGB inverted (BGR), making it green 
                                                         # 2 is the tickness of the rectangle
    cv2.imshow('Face Detection', frame) 
    cv2.waitKey(1) # auto refreshs the image of the webcan every 1 ms
