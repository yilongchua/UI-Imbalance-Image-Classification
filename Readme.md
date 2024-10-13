### UI pattern Image Classification Problem

## Context:
Commonly occurring UI patterns in design and label them so the appropriate code can be generated
Please note, this will require some data engineering as the None class is not labelled.

## Problem:
Build a Image classification model to classify a given bounding box
1. Grid - https://www.w3schools.com/css/css_grid.asp 
2. Popup - https://www.w3schools.com/howto/howto_js_popup_form.asp
3. Progress Bar - https://www.w3schools.com/tags/tag_progress.asp
4. None - Does not belong either of the above

## Data:
* see data folder "Locofy_data.json"

## Approach to maximize f1 score 'macro' for evaluation as this task is mainly focus on imbalance dataset
1. Create None Dataset 
    * 1st Iteration: Random crop based on 25% - 75% of width/height and starting position
    * 2nd Iteration: Use adaptive thresholding (binary inverse) to find lcoation of interest and crop out largest area to use for none class
2. Check if its possible to increase other popup and Progress Bar
    * 2nd Iteration: Randomly swapping rgb layers, introducing hue and saturation adjustments
3. Consider data augmentation process focusing on different techniques for different classes
    * progress bar & slider => pad to square 
    * different classes have different transformation
4. Change weights for CrossEntropyLoss based on number of image per dataset
5. metrics : f1, recall, acc. 
    * Likely to use f1 'macro' for evaluation.
6. Attempt to use partial freezing of model weights focus on layer4
    * Maintain first 3 layers of ResNet50
        * Results (Freezing)
            * val Loss: 0.4261 Acc: 0.8596, f1:0.8070, recall:0.8634
        * Results (No Freezing)
            * val Loss: 0.2752 Acc: 0.8924, f1:0.8595, recall:0.8815 

## Model used.
1. Using Resnet50 as main model.
2. Please download pretrained model from link: https://drive.google.com/drive/folders/11oieHjyCdPLzjBS-flhyuL8_JjV4MKnK?usp=sharing

# Current Solution

## Future Improvements (future Iterations)
* EfficientNet, Inceptionv3 could potential be used.
* Focal Loss Function can be used at the last layer of Resnet50 as it is design for imbalanced datasets
* Docker file and flask API for test 
* quantising Model 
* improve post processing using original image extracted location 
    1. Meta-data like location of boundbox in reference to the entire image (Multi-input information)
    2. Image crop out width and height 

## Inference script
Please use inference.py and test directly.
type into terminal : python infer.py -i "./test_image/none_26_75777.png" -d "cpu"