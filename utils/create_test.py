

import os 
import cv2


if __name__=="__main__":
    camera = 4
    time = 'N'
    frame = 4
    
    train_data = "/home/gb2t/Desktop/huynho/Co-DETR/data"
    test_data = "/home/gb2t/Desktop/huynho/Co-DETR/testset"
    output_data = "/home/gb2t/Desktop/AIO_Pending/test_images"
    
    if camera in [1,2,4,7]:
        image_path = f"{train_data}/test/images/camera{camera}_{time}_{frame}.png"
    elif camera < 19:
        image_path = f"{train_data}/train/images/camera{camera}_{time}_{frame}.png"
    else:
        image_path = f"{test_data}/images/camera{camera}_{time}_{frame}.png"
    
    image = cv2.imread(image_path)
    cv2.imwrite(f"{output_data}/camera{camera}_{time}_{frame}.png",image)