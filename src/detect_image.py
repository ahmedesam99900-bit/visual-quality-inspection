from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"
IMAGE_PATH = "data/bottles_test.jpg"
BOTTLE_CLASS_ID = 39


# Load pretrained YOLO model
model = YOLO(MODEL_PATH)

# Run bottle detection
results = model(
    IMAGE_PATH,
    classes=[BOTTLE_CLASS_ID]
)

# Get detections
boxes = results[0].boxes

print(f"\nBottles detected: {len(boxes)}")

# Display annotated image
results[0].show()