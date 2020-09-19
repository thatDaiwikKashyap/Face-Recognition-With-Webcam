#Call In CV2 https://pypi.org/project/opencv-python/
import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Video cap:
webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225 , 0), 3)

    cv2 .imshow('Daiwik Face Detector' , frame)
    cv2.waitKey(1)
