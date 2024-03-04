

# data_dir="/home/gb2t/Desktop/AIO_Pending/Fisheye8K"
data_dir="/home/gb2t/Desktop/AIO_Pending/Fisheye1K"
mask_dir="/home/gb2t/Desktop/AIO_Pending/masks"

python utils/create_fisheye_mask.py --data_dir "$data_dir" --mask_dir "$mask_dir"