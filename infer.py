import torch, os, json, argparse
import torchvision.transforms as transforms
from PIL import Image
from src import *

def main(args):
    if args.device !='cpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'{device=}')
    else:
        device = torch.device('cpu')
        print(f'Using CPU')
    class_names = {0: 'grid', 1: 'none', 2: 'popup',  3: 'progress_bar'}

    model = load_model('./model_weights/best_model_v3.pth', device)    
    image_tensor = preprocess_image(args.image_path, device)
    predicted_class = predict(model, image_tensor, class_names)
    print(f'Predicted class: {predicted_class}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classification Inference')
    parser.add_argument('-i', '--image-path', type=str, required=True, help='Path to the image file for inference')
    parser.add_argument('-d', '--device', type=str, help='device use for inference, cuda, cpu')
    
    args = parser.parse_args()
    main(args)
    