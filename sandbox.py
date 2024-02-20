import cv2
import numpy as np

# Global variables
selected_point = (-1, -1)
contours = []

def draw_contours(image, contours):
    # Draw contours on the image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

def mouse_callback(event, x, y, flags, param):
    global selected_point

    if event == cv2.EVENT_LBUTTONDOWN:
        # When the left mouse button is clicked, save the coordinates
        selected_point = (x, y)
        print("Selected Point:", selected_point)

        # Check if the selected point is inside any of the contours
        for contour in contours:
            if cv2.pointPolygonTest(contour, selected_point, False) >= 0:
                # User-selected point is inside this contour
                draw_contours(image, [cv2.approxPolyDP(contour, 0.02*cv2.arcLength(contour, True), True)])

        cv2.imshow("Contour Image", image)

# Read the image
image_path = "sampleImages/bouldering_wall_1.webp"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary image
ret, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Create a window and set the callback function for mouse events
cv2.namedWindow("Contour Image")
cv2.setMouseCallback("Contour Image", mouse_callback)

while True:
    # Display the image
    cv2.imshow("Contour Image", image)

    # Check for the 'ESC' key to exit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Release resources
cv2.destroyAllWindows()
