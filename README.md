# facemask_detection
This contains the code and weight for facemask detection. the weight was trained at epoch 300.
The original code can be downloaded from https://github.com/ultralytics/yolov5

The code in my program is modified to download image encoded to base64 string from firebase and decode it to image file(PNG) and store it in the input folder.
The detect from automaticaly takes this file from input folder and will make necessary changes in firebase.
