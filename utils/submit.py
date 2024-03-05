
import argparse
import os
import json
from tqdm import tqdm
import cv2

def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    # img_name = img_name[:-4]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
    return imageId


def yolo_to_target_bbox(yolo_label, img_width, img_height):
    """
    Convert YOLO label to target bounding box format [x1, y1, width, height].

    Args:
    - yolo_label (list): YOLO label containing [class_id, x_center, y_center, width, height]
    - img_width (int): Width of the image
    - img_height (int): Height of the image

    Returns:
    - target_bbox (tuple): Target bounding box in [x1, y1, width, height] format.
    """
    x_center = float(yolo_label[1])
    y_center = float(yolo_label[2])
    width = float(yolo_label[3])
    height = float(yolo_label[4])
    
    # Convert relative coordinates to absolute pixel coordinates
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    # Calculate absolute bounding box coordinates
    xmin = int(x_center - width / 2)
    ymin = int(y_center - height / 2)
    xmax = int(x_center + width / 2)
    ymax = int(y_center + height / 2)

    # Calculate target bounding box format [x1, y1, width, height]
    target_x1 = xmin
    target_y1 = ymin
    target_width = width
    target_height = height

    return (target_x1, target_y1, target_width, target_height)


def main(args):
    image_dir = args.image_dir
    label_dir = args.label_dir
    submit_dir = args.submit_dir
    os.makedirs(submit_dir,exist_ok=True)
    file_names = os.listdir(label_dir)
    submit = []
    for label_name in tqdm(file_names,desc="Reading labels"):
        label_path = os.path.join(label_dir,label_name)
        
        image_name = label_name.replace('txt','png')
        image_path = os.path.join(image_dir,image_name)
        image_id = get_image_Id(image_name)
        
        image = cv2.imread(image_path)
        h,w,_=image.shape
        with open(label_path,'r') as file:
            for line in file:
                # Split the line into individual values
                values = line.split()
                cate_id = values[0]
                score = values[-1]
                bbox = yolo_to_target_bbox(values,w,h)
                submit.append({
                    "image_id": int(image_id),
                    "category_id": int(cate_id),
                    "bbox": bbox,
                    "score": float(score)
                })
    json_path = os.path.join(submit_dir,f"{args.name}.json")
    with open(json_path,'w') as json_file:
        json.dump(submit, json_file, indent=4)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",type=str)
    parser.add_argument("--label_dir",type=str)
    parser.add_argument("--submit_dir",type=str,default="submission")
    parser.add_argument("--name",type=str,default="submission")
    args = parser.parse_args()
    main(args)