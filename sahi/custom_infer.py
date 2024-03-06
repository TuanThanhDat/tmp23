from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
import os

def main():
    root_dir = os.getcwd()
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov9',
        confidence_threshold=0.3,
        device="cuda", # or 'cuda:0'
    )

if __name__=="__main__":
    main()