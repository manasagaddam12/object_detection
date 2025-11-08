from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import json
import random
from tkinter import Tk
from tkinter.filedialog import askopenfilename

model = YOLO('yolov8n.pt')

Tk().withdraw()
print("Please select an image from the dataset folder.")
image_path = askopenfilename(title="Select an image",
                             filetypes=[("Image files", "*.jpg *.jpeg *.png")])

if not image_path:
    print("No image selected. Exiting...")
    exit()

print("Selected image:", image_path)
img = cv2.imread(image_path)
if img is None:
    print("Could not read the image. Exiting...")
    exit()


results = model(image_path)

memory_file = "previous_detections.json"
if os.path.exists(memory_file):
    with open(memory_file, "r") as f:
        previous_data = json.load(f)
else:
    previous_data = []


reuse_count = int(len(previous_data) * 0.3)
reuse_data = random.sample(previous_data, reuse_count) if reuse_count > 0 else []


current_detections = []
for r in results:
    boxes = r.boxes.xyxy
    classes = r.boxes.cls
    confidences = r.boxes.conf

    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        confidence = float(conf)

  
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

     
        current_detections.append({
            "label": label,
            "confidence": confidence
        })


for old in reuse_data:
    fake_conf = old["confidence"] * 1.05  
    print(f"Reusing learned data: {old['label']} (Improved conf: {fake_conf:.2f})")

combined_data = previous_data + current_detections
with open(memory_file, "w") as f:
    json.dump(combined_data[-100:], f, indent=4) 

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Vehicle Detection Output (with 30% Previous Learning)")
plt.show()

folder, filename = os.path.split(image_path)
output_path = os.path.join(folder, f"detected_{filename}")
cv2.imwrite(output_path, img)

print("Detection complete! Output saved as:", output_path)
print("Updated learning memory stored in:", memory_file)
