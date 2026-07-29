import cv2
import numpy as np

# Create a simple test image (blue rectangle)
img = np.zeros((480, 640, 3), dtype=np.uint8)
img[:] = (255, 0, 0)  # blue in BGR
cv2.putText(img, 'TEST WINDOW - press q to close', (30, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

cv2.imshow('Test Window', img)
print("Window should be showing now. Press 'q' in the window to close.")

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Closed.")