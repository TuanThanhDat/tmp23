
# weight="/home/gb2t/Desktop/AIO_Pending/yolov9/weights/best.pt"
weight="/home/gb2t/Desktop/Gysby_3712/YOLOv9/yolov9/runs/train/exp3/weights/best.pt"
data_dir="/home/gb2t/Desktop/AIO_Pending/preprocess_images"
# data_dir="/home/gb2t/Desktop/huynho/Co-DETR/testset/images"

GPU=2

rm -r /home/gb2t/Desktop/AIO_Pending/yolov9/runs
# CONDA_ACT="source /home/$USER/anaconda3/bin/activate"

# mask
# $CONDA_ACT tuan_SAHI
CUDA_VISIBLE_DEVICES=3 python yolov9/detect.py --weights "$weight" --source "$data_dir" --device 3 --save-txt --save-conf --img 1280

