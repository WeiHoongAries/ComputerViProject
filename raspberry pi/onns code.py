import onnxruntime as rt
import cv2 as cv
import numpy as np
from PIL import Image,ImageDraw

dataset = ["raspberry pi\\pen15.jpg", "raspberry pi\\222.jpg"]

def preprocessing(image):
    size = 640
    #height,width,channel
    height,width = image.shape[:2]
    aspect_ratio = width/height
    if width > height:
        new_width = size
        new_height = int(new_width/aspect_ratio)
        image=cv.resize(image,(new_width,new_height))
        padding = size - new_height
        image=cv.copyMakeBorder(image,0,padding,0,0,cv.BORDER_CONSTANT)
    else:
        new_height=size
        new_width = int(aspect_ratio * new_height)
        image=cv.resize(image,(new_width,new_height))
        padding = size - new_width
        image=cv.copyMakeBorder(image,0,0,0,padding,cv.BORDER_CONSTANT)
    # cv.imshow("Prerprocessed",image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return image

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

def AI_inference(preprocessed_image):
    sess = rt.InferenceSession("raspberry pi\\weights\\best.onnx",providers=['CPUExecutionProvider'])
    output0 = sess.run(None,{"images":preprocessed_image})
    return output0

def crop(image,bbox,preprocessed_image):
    size=640
    original_height,original_width=image.shape[:2]
    aspect_ratio = original_width/original_height
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    
    if original_width > original_height:
        new_width = size
        new_height = int(new_width / aspect_ratio)
        padding = size - new_height
        # Remove padding
        no_pad_image = preprocessed_image[:new_height, :]
    else:
        new_height = size
        new_width = int(aspect_ratio * new_height)
        padding = size - new_width
        # Remove padding
        no_pad_image = preprocessed_image[:, :new_width]

    resized_height_scale = original_height/no_pad_image.shape[0]
    resized_width_scale = original_width/no_pad_image.shape[1]
    x1=int(x1*resized_width_scale)
    y1=int(y1*resized_height_scale)
    x2=int(x2*resized_width_scale)
    y2=int(y2*resized_height_scale)
    image = image[y1:y2,x1:x2]
    
    return image



    return cropped_image

def main():
    for file in dataset:
        image = cv.imread(file,cv.IMREAD_COLOR) #BGR
        preprocessed_image = preprocessing(image)
        #opencv to PIL format
        RGB_image = Image.fromarray(cv.cvtColor(preprocessed_image,cv.COLOR_BGR2RGB))
        #RGB_Image as draw target
        draw =ImageDraw.Draw(RGB_image)
        
        
        #pixel values convert to float format and normalized to 0-1 range 
        normalized = preprocessed_image.astype(np.float32) / 255
        #H(0),W(1),C(2) to CHW
        transposed = np.transpose(normalized,[2,0,1])
        expand = np.expand_dims(transposed,axis=0)
        output0 = AI_inference(expand)
        output0 = np.array(output0[0])
        bbox, confidences = postprocess(output0)
        for boxes in bbox:
            
            cropped_image = crop(image,boxes,preprocessed_image)
            # crop = preprocessed_image[y1:y2,x1:x2]
            # draw.rectangle(boxes,outline='red',width=2)
            # image = np.array(RGB_image)
            # image = cv.cvtColor(image,cv.COLOR_RGB2BGR)

            cv.imshow("Result",cropped_image)
            cv.waitKey(0)
            cv.destroyAllWindows()

if __name__ == "__main__":
    main()