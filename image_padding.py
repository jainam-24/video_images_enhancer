import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image_path = 'diamond_input_greayish_back.jpg'
img = cv2.imread(image_path)
img_pil=Image.open("C:/Users/JAINAM/Downloads/d360/image_video_enhancer/input_images/bA1.jpg")
# Apply reflection padding (50 pixels on all sides as an example)
top, bottom, left, right = 60, 60, 60, 60
padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)

# Display the padded image
plt.imshow(cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide the axis

padded_img=cv2.resize(padded_img,(635,635))
# Optionally save the image
cv2.imwrite('padded_image_now.jpg', padded_img)
