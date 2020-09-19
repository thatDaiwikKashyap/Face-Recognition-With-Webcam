import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detectfaces in
#img = cv2.imread('quad.jpg')
img = cv2.imread('pic.jpg')
#img = cv2.imread('pic2.jpg')

# Must Convert to grey scale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Face
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0 , 255), 3)

#print(face_coordinates)


cv2 .imshow('Daiwik Face Detector' , img)
cv2.waitKey()

print("Hello World")