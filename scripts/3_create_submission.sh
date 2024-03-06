

label_dir="/home/gb2t/Desktop/AIO_Pending/yolov9/runs/detect/exp/labels"
image_dir="/home/gb2t/Desktop/AIO_Pending/preprocess_images"
# image_dir="/home/gb2t/Desktop/huynho/Co-DETR/testset/images"

USER="gb2t"
CONDA_ACT="source /home/$USER/anaconda3/bin/activate"

$CONDA_ACT tuan_FFT
python utils/submit.py --image_dir "$image_dir" --label_dir "$label_dir" --name "submission_4_DIP_new"