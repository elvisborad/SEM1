import cv2
import numpy as np

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the webcam
cap = cv2.VideoCapture(0)

# Choose filter mode
mode = 1  # 1 = Blur, 2 = Cartoon, 3 = Color Invert

print("Press '1' for Blur, '2' for Cartoon, '3' for Invert, 'q' to Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        if mode == 1:  # Blur
            filtered = cv2.GaussianBlur(face_roi, (55, 55), 30)
        elif mode == 2:  # Cartoon
            gray_face = cv2.medianBlur(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY), 7)
            edges = cv2.adaptiveThreshold(gray_face, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(face_roi, 9, 300, 300)
            filtered = cv2.bitwise_and(color, color, mask=edges)
        elif mode == 3:  # Invert colors
            filtered = cv2.bitwise_not(face_roi)
        else:
            filtered = face_roi

        frame[y:y+h, x:x+w] = filtered
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("AI Face Filter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        mode = 1
    elif key == ord('2'):
        mode = 2
    elif key == ord('3'):
        mode = 3

cap.release()
cv2.destroyAllWindows()
