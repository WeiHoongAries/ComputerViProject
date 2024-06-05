# License Plate Detection on Raspberry Pi
This project demonstrates how to detect and extract license plates from images using a pre-trained ONNX model on a Raspberry Pi. The model is loaded and run using the ONNX Runtime, and the preprocessing and postprocessing steps are handled with OpenCV and NumPy.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- ONNX Runtime
- PIL (Pillow)

## Installation
Install Python dependencies:
```
pip install opencv-python-headless numpy onnxruntime pillow
```
## Download the ONNX model:
Ensure you have the best.onnx model file saved in the raspberry pi/weights directory.

## Usage
Preprocessing Function
This function resizes and pads the input image to a fixed size of 640x640 while maintaining the aspect ratio.
```
def preprocessing(image):
    size = 640
    height, width = image.shape[:2]
    aspect_ratio = width / height
    if width > height:
        new_width = size
        new_height = int(new_width / aspect_ratio)
        image = cv.resize(image, (new_width, new_height))
        padding = size - new_height
        image = cv.copyMakeBorder(image, 0, padding, 0, 0, cv.BORDER_REPLICATE)
    else:
        new_height = size
        new_width = int(aspect_ratio * new_height)
        image = cv.resize(image, (new_width, new_height))
        padding = size - new_width
        image = cv.copyMakeBorder(image, 0, 0, 0, padding, cv.BORDER_REPLICATE)
    cv.imshow("Preprocessed", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return image
```
    
## Postprocessing Function
This function processes the output tensor from the model, applies confidence thresholding, and performs Non-Maximum Suppression (NMS) to filter out redundant bounding boxes.
```
def postprocess(output_tensor, conf_threshold=0.5, nms_threshold=0.4):
    # Reshape the output tensor to (5, 8400)
    output_tensor = output_tensor.reshape(5, 8400)
    
    # Extract the bounding box coordinates and confidence scores
    boxes = output_tensor[:4, :].transpose()  # shape (8400, 4)
    confidences = output_tensor[4, :]  # shape (8400,)

    # Apply a confidence threshold
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]

    # Convert boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    # Apply Non-Maximum Suppression (NMS)
    indices = cv.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_threshold, nms_threshold)

    return boxes[indices], confidences[indices]
```
    
## Main Function
This function loads the ONNX model, preprocesses the images, runs the model to get detections, and then crops and displays the detected license plate areas.
```
def main():
    sess = rt.InferenceSession("raspberry pi\\weights\\best.onnx", providers=['CPUExecutionProvider'])
    for file in dataset:
        image = cv.imread(file, cv.IMREAD_COLOR)
        preprocessed_image = preprocessing(image)

        RGB_image = Image.fromarray(cv.cvtColor(preprocessed_image, cv.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(RGB_image)
        
        # nchw expand, transpose
        normalized = preprocessed_image.astype(np.float32) / 255
        transposed = np.transpose(normalized, [2, 0, 1])
        expand = np.expand_dims(transposed, axis=0)
        output0 = sess.run(None, {"images": expand})
        output0 = np.array(output0[0])
        boxes, confidences = postprocess(output0)
        
        x1 = int(boxes[0, 0])
        y1 = int(boxes[0, 1])
        x2 = int(boxes[0, 2])
        y2 = int(boxes[0, 3])
        crop = preprocessed_image[y1:y2, x1:x2]
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        image = np.array(RGB_image)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        cv.imshow("Result", crop)
        cv.waitKey(0)
        cv.destroyAllWindows()
```        
## How to Run
Ensure your ONNX model file is located at raspberry pi/weights/best.onnx.
Place your input images in the dataset list.

## Run the script:
python your_script_name.py
