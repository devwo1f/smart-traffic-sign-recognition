import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('best.pt')

# Load an example image (let's use one from the data preparation if it exists, or just open a webcam feed briefly to test)
# Since we don't have a guaranteed test image path, let's grab one from the Mapillary dataset that was extracted, or just run a quick webcam test.

print("Model loaded successfully!")
print(f"Model classes: {len(model.names)}")

# To test inference, uncomment below if you have a test.jpg
# results = model('test.jpg') 
# results[0].show()  # Display to screen

