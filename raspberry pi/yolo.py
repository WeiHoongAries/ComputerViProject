from ultralytics import YOLO
model = YOLO("raspberry pi\\weights\\best.pt")

print(model.info)