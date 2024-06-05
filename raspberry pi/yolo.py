from ultralytics import YOLO
model = YOLO("raspberry pi\\weights\\best.onnx")
results = model("raspberry pi\\pen15.jpg")
for result in results:
    result.show()
print(model.info)