import os, torch
import numpy as np
from torchvision import  transforms, models
from PIL import Image

base_transform = [transforms.Resize((224, 224)),  
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

def non_max_suppression(boxes, iou_threshold):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)    
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Compute IoU between the kept box and the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return boxes[keep].tolist()

def pad_to_square(image):
    w, h = image.size  
    if w/h > 10 or h/w > 10:  
        padding = (0, (w - h) // 2, 0, (w - h) - (w - h) // 2) 
        return transforms.functional.pad(image, padding, fill=0, padding_mode='constant')
    else:  
        return image
        
def load_model(model_path, device):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)  

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)
    model.eval()  
    return model

def preprocess_image(image_path, device):
    preprocess = transforms.Compose([transforms.Lambda(pad_to_square)] + base_transform)
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    return image.to(device)

def predict(model, image_tensor, class_names):
    with torch.no_grad(): 
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds.item()]
    return predicted_class


 
