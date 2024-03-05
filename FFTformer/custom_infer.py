import os
import torch
import argparse
from basicsr.models.archs.fftformer_arch import  fftformer
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image as Image
from tqdm import tqdm

import cv2
import numpy as np

def resize(image,nw,nh):
    return cv2.resize(image,(int(nw),int(nh)))

class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        
        items = os.listdir(image_dir)
        if ('train' in items) and ('test' in items):
            self.image_list = sorted(os.listdir(os.path.join(image_dir, 'train/images')))
            self.image_list = self.image_list + sorted(os.listdir(os.path.join(image_dir, 'test/images')))
            self.dataset_type = "train"
        elif ('images' in items):
            self.image_list = sorted(os.listdir(os.path.join(image_dir, 'images')))
            self.dataset_type = "eval"
        else:
            self.image_list = sorted(os.listdir(image_dir))
            self.dataset_type = "test"
        # self.image_list = os.listdir(os.path.join(image_dir, 'input/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        name = self.image_list[idx]
        if self.dataset_type=='train':
            camera = int(name.split('_')[0].replace('camera',''))
            if camera in [1,2,4,7]:
                image = cv2.imread(os.path.join(self.image_dir,'test/images',name))
            else:
                image = cv2.imread(os.path.join(self.image_dir,'train/images',name))
        elif self.dataset_type=='eval':
            image = cv2.imread(os.path.join(self.image_dir,'images',name))
        else:
            image = cv2.imread(os.path.join(self.image_dir,name))
            
        # label = Image.open(os.path.join(self.image_dir, 'target', self.image_list[idx]))
        ih,iw = image.shape[:2]
        if ih*iw > 1200*1200:
            image = resize(image,1200,1200)
        image = F.to_tensor(image)
        # name = os.path.split(self.image_list[idx])[-1]
        return (image, (iw,ih), name)

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label
        
    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = os.path.split(x)[-1].split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError


def create_dataloader(image_dir, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(image_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    return dataloader


def main(args):
    # CUDNN
    # cudnn.benchmark = True
    #
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = fftformer()
    # print(model)
    if torch.cuda.is_available():
        model.cuda()
    _eval(model, args)


def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict,strict = True)
    device = torch.device( 'cuda')
    dataloader = create_dataloader(args.data_dir)
    torch.cuda.empty_cache()

    model.eval()

    with torch.no_grad():
        # Main Evaluation
        for data in tqdm(dataloader,desc='Applying FFT'):
            input_img, origin_shape, name = data
            name = name[0]

            input_img = input_img.to(device)
            b, c, h, w = input_img.shape
            h_n = (32 - h % 32) % 32
            w_n = (32 - w % 32) % 32
            input_img = torch.nn.functional.pad(input_img, (0, w_n, 0, h_n), mode='reflect')

            pred = model(input_img)
            torch.cuda.synchronize()
            pred = pred[:, :, :h, :w]
            pred_clip = torch.clamp(pred, 0, 1)

            if args.save_image:
                pred_np = pred_clip.squeeze(0).cpu().numpy()
                pred_np = (pred_np * 255).astype(np.uint8)
                pred_np = pred_np.transpose(1,2,0)
                if (w,h) != origin_shape:
                    pred_np = resize(pred_np, origin_shape[0], origin_shape[1])
                    pred_np = pred_np[:,:,::-1]
                pred_pil = Image.fromarray(pred_np, 'RGB')
                save_name = os.path.join(args.output_dir,name)
                pred_pil.save(save_name)

if __name__ == '__main__':
    fft_dir = os.path.join(os.getcwd(),'FFTformer')
    
    parser = argparse.ArgumentParser()
    # Directories
    parser.add_argument('--model_name', default='fftformer', type=str)
    parser.add_argument('--data_dir', type=str, default='/media/kls/新加卷1/CVPR_2023/GoPro_test/')
    parser.add_argument('--output_dir', type=str, default='/media/kls/新加卷1/CVPR_2023/GoPro_test/')

    # Test
    parser.add_argument('--test_model', type=str, default=f'{fft_dir}/pretrain_model/net_g_Realblur_J.pth')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

    args = parser.parse_args()
    main(args)
