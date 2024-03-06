import os
import cv2
import numpy as np 
import argparse
import glob
from tqdm import tqdm

def apply_fisheye_mask(image, mask):
    mask = mask.astype(np.int8)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def main(args):
    data_dir = args.data_dir
    mask_dir = args.mask_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    items = os.listdir(data_dir)
    if ('test' in items) and ('train' in items):
        path_list = sorted(glob.glob(f"{data_dir}/train/images/*.png"))
        path_list = path_list + sorted(glob.glob(f"{data_dir}/test/images/*.png"))
    elif "images" in items:
        path_list = sorted(glob.glob(f"{data_dir}/images/*.png"))
    else:
        path_list = sorted(glob.glob(f"{data_dir}/*.png"))

    for path in tqdm(path_list, desc="Applying mask"):
        image_name = os.path.split(path)[-1]
        image = cv2.imread(path)
        camera = image_name.split('_')[0][6:]
        time = image_name.split('_')[1]
        mask_list = sorted(glob.glob(f'{mask_dir}/camera{camera}_{time}_*.png'))
        for mask_path in mask_list:
            mask = cv2.imread(mask_path)
            if mask.shape != image.shape:
                continue
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            result = apply_fisheye_mask(image,mask)
            cv2.imwrite(f'{output_dir}/{image_name}',result)
        # print(f"==Applied mask to {image_name} successfully!!!")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str)
    parser.add_argument("--mask_dir",type=str)
    parser.add_argument("--output_dir",type=str)
    args = parser.parse_args()
    main(args)