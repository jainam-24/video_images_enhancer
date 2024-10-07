
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'C:/Users/JAINAM/Downloads/d360/image_video_enhancer/bA1.jpg'
img = cv2.imread(image_path)

# Apply constant padding (for simplicity) which will be filled in later
top, bottom, left, right = 100, 100, 100, 100
padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# Create a mask where the padded area will be inpainted (1 for areas to be filled)
mask = np.zeros((padded_img.shape[0], padded_img.shape[1]), dtype=np.uint8)
mask[:top, :] = 1  # Top padding
mask[-bottom:, :] = 1  # Bottom padding
mask[:, :left] = 1  # Left padding
mask[:, -right:] = 1  # Right padding

# Apply inpainting to fill the padded areas based on the original image
inpainted_img = cv2.inpaint(padded_img, mask, 3, cv2.INPAINT_TELEA)

# Save the inpainted image
cv2.imwrite('content_aware_padded_image.jpg', inpainted_img)

# Display the inpainted image
plt.imshow(cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
