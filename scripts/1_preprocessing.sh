

# data_dir="/home/gb2t/Desktop/AIO_Pending/Fisheye8K"
data_dir="/home/gb2t/Desktop/AIO_Pending/test_images"
mask_dir="/home/gb2t/Desktop/AIO_Pending/masks"
output_dir="/home/gb2t/Desktop/AIO_Pending/preprocess_images"
GPU=2

python utils/apply_mask.py --data_dir "$data_dir" --mask_dir "$mask_dir" --output_dir "$output_dir"
CUDA_VISIBLE_DEVICES=$GPU python Image-Adaptive-YOLO/custom_infer.py --data_dir "$output_dir" --output_dir "$output_dir"
# CUDA_VISIBLE_DEVICES=$GPU 
# python Image-Adaptive-YOLO/evaluate.py --use_gpu 0