

from pathlib import Path
import os

FILE = Path(__file__).resolve()
# print(FILE.parents[0])
ROOT = FILE.parents[0]
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# print(ROOT)

import torch
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
# from utils.plots import Annotator, colors, save_one_box
# from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import (letterbox)
# import cv2
import numpy as np

class YOLOV9:
    def __init__(self,
                 weight_path, 
                 device, 
                 yml_path,
                 imgsz):
        self.model = DetectMultiBackend(weights=weight_path,device=device,dnn=False,data=yml_path,fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.device = device
        
    def predict(self, im):
        '''
        im: np.array, RGB?
        '''
        im = letterbox(im, self.imgsz, stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        
        pred = self.model(im,augment=False,visualize=False)
        return pred
    
if __name__=="__main__":
    weight_path = "/home/gb2t/Desktop/Gysby_3712/YOLOv9/yolov9/runs/train/exp3/weights/best.pt"
    yml_path = "/home/gb2t/Desktop/Gysby_3712/YOLOv9/data.yaml"
    device = torch.device('cpu')
    imgsz = 1280
    
    model = YOLOV9(\
        weight_path=weight_path,\
        device=device,\
        yml_path=yml_path,\
        imgsz=imgsz)
    
    img_path = "/home/gb2t/Desktop/AIO_Pending/test_images/camera1_A_10.png"
    img = cv2.imread(img_path)
    pred = model.predict(img)
    pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
    print(len(pred[0,0]))