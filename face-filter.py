import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the face detection model from dlib
face_detector = dlib.get_frontal_face_detector()

# Load the user's image
image_path = 'user_image.jpg'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_detector(gray_image)

# Ensure a face is detected
if len(faces) > 0:
    # Get the coordinates of the first detected face
    face = faces[0]
    top, right, bottom, left = face.top(), face.right(), face.bottom(), face.left()

    # Load and resize the virtual glasses image
    glasses_image = cv2.imread('glass_image.jpg')
    glasses_height = bottom - top
    glasses_width = right - left
    glasses_image = cv2.resize(glasses_image, (glasses_width, glasses_height))

    # Apply the virtual glasses to the user's face
    for i in range(glasses_height):
        for j in range(glasses_width):
            if glasses_image[i, j].any() < 235:  # Avoid white pixels
                image[top + i, left + j] = glasses_image[i, j]

    # Display the modified image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
else:
    print("No face detected in the image")