from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train3/weights/best.pt")

# Customize validation settings
validation_results = model.val(data="/home/tarun/Desktop/antcam/datasets/ants_manual_annotation/ants_manual_annotation.yaml", imgsz=1920, batch=1, conf=0.362, iou=0.7, device="cpu", max_det=1000)

print ('map50:', validation_results.box.map50)
print ('f1 score:', validation_results.box.f1)
