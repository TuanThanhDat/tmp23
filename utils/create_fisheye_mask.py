import cv2
import numpy as np 
import os
import argparse
import glob


def get_fisheye_mask(img, threshold=20, equalizeHist=False):
    img_shape = img.shape
    img_center = (img_shape[1] // 2, img_shape[0] // 2)

    # convert RGB to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if equalizeHist:
        gray = cv2.equalizeHist(gray)
    
    # blurring to remove noise
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.medianBlur(gray, 7)
    
    # thresholding image
    _, thresh8 = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Getting edge of threshold image
    mask = cv2.adaptiveThreshold(thresh8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)
    mask = cv2.bitwise_not(mask)
    
    # Fill color at center of image to remove noise outside fisheye
    final_mask = mask.copy()
    cv2.floodFill(mask, None, img_center, 255)
    final_mask = mask - final_mask
    
    # Fill color at 4 corners of image to remove noise within fisheye
    mask = final_mask.copy()
    h, w = mask.shape[:2]
    final_mask = np.zeros_like(mask)
    delta = 20
    corners = [(0,0),(0,w-1),(h-1,0),(h-1,w-1),\
                (delta,0),(delta,w-1),(h-delta-1,0),(h-delta-1,w-1),\
                (0,delta),(0,w-delta-1),(h-1,delta),(h-1,w-delta-1)]
    for y, x in corners:
        if mask[y,x] == 0:
            mask_copy = mask.copy()
            cv2.floodFill(mask_copy, None, (x,y), 255)
            floodfill_area = ((mask_copy-mask) == 255).astype(np.uint8)
            final_mask = cv2.bitwise_or(final_mask, floodfill_area)

    # convert 1 to 255
    final_mask = final_mask*255
    final_mask = cv2.bitwise_not(final_mask)
    return final_mask


def get_path_list(data_dir,camera_id,time_id,data_type='train'):
    if data_type=='train':
        if camera_id in [1,2,4,7]:
            return sorted(glob.glob(f"{data_dir}/test/images/camera{camera_id}_{time_id}_*.png"))
        else:
            return sorted(glob.glob(f"{data_dir}/train/images/camera{camera_id}_{time_id}_*.png"))
    else:
        return sorted(glob.glob(f"{data_dir}/images/camera{camera_id}_{time_id}_*.png"))


def main(args):
    data_dir = args.data_dir
    mask_dir = args.mask_dir
    os.makedirs(mask_dir,exist_ok=True)
    
    threshold_list = [
        (1,'A',20),
        (2,'A',10),
        (3,'A',20),
        (3,'N',40),
        (4,'M',70),
        (4,'A',30),
        (4,'E',40),
        (4,'N',30),
        (5,'A',25),
        (6,'A',20),
        (7,'A',25),
        (8,'A',25),
        (9,'A',25),
        (10,'A',25),
        (11,'M',60),
        (12,'A',15),
        (13,'A',10),
        (14,'A',40),
        (15,'A',15),
        (16,'A',10),
        (17,'A',15),
        (18,'A',60),
        (19,'A',10),
        (20,'A',30),
        (21,'A',20),
        (22,'A',20),
        (23,'A',10),
        (24,'A',30),
        (25,'A',20),
        (26,'A',20),
        (27,'A',15),
        (28,'A',30),
        (29,'A',20),
        (29,'N',5)
    ]
    
    items = os.listdir(data_dir)
    if ('test' in items) and ('train' in items):
        data_type = "train"
    else:
        data_type = "eval"
    sublist = threshold_list[:22] if data_type=='train' else threshold_list[22:]
    
    for camera,time,threshold in sublist:
        path_list = get_path_list(data_dir,camera,time,data_type)
        list_shape = []
        for path in path_list:
            image = cv2.imread(path)
            image_name = os.path.split(path)[-1]
            if len(list_shape) != 0 and image.shape in list_shape:
                continue
            list_shape.append(image.shape)
            mask = get_fisheye_mask(image,threshold,time=='N')
            cv2.imwrite(f"{mask_dir}/{image_name}",mask)
            print(f"==Created mask for {image_name} successfully!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str)
    parser.add_argument("--mask_dir",type=str)
    
    args = parser.parse_args()
    print("Start create mask")
    main(args)