from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict

import os

def main():
    root_dir = os.getcwd()
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov9',
        model_path="/home/gb2t/Desktop/Gysby_3712/YOLOv9/yolov9/runs/train/exp3/weights/best.pt",
        confidence_threshold=0.1,
        device="cuda", # or 'cuda:0'
        image_size=1280
    )
    
    image_path = "/home/gb2t/Desktop/AIO_Pending/test_images/camera3_A_4.png"
    result = get_prediction(image_path,detection_model)
    
    result.export_visuals(export_dir="demo_data/")
    # Image("demo_data/prediction_visual.png")

if __name__=="__main__":
    main()