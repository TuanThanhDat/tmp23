

# data_dir="/home/gb2t/Desktop/huynho/Co-DETR/data"
data_dir="/home/gb2t/Desktop/huynho/Co-DETR/testset"
# data_dir="/home/gb2t/Desktop/AIO_Pending/test_images"
mask_dir="/home/gb2t/Desktop/AIO_Pending/masks"
output_dir="/home/gb2t/Desktop/AIO_Pending/preprocess_images"
GPU=2

USER='gb2t'
CONDA_ACT="source /home/$USER/anaconda3/bin/activate"

rm -r "$output_dir"

# mask
$CONDA_ACT tuan_FFT
python utils/apply_mask.py --data_dir "$data_dir" --mask_dir "$mask_dir" --output_dir "$output_dir"

# FFT module
CUDA_VISIBLE_DEVICES=$GPU python FFTformer/custom_infer.py --data_dir "$output_dir" --output_dir "$output_dir"
# conda deactivate

# # DIP module
# $CONDA_ACT tuan_DIP
# CUDA_VISIBLE_DEVICES=$GPU python Image-Adaptive-YOLO/custom_infer.py --data_dir "$output_dir" --output_dir "$output_dir"

# # mask
python utils/apply_mask.py --data_dir "$output_dir" --mask_dir "$mask_dir" --output_dir "$output_dir"
conda deactivate