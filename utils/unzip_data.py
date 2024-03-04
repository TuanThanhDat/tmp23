import zipfile
import argparse
import os
from tqdm import tqdm

def unzip_file(zip_path, des_path):
    with zipfile.ZipFile(zip_path,'r') as zip_ref:
        file_list = zip_ref.namelist()
        for file in tqdm(file_list,desc="Extracting files",unit="files"):
            zip_ref.extract(file, des_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path",type=str)
    parser.add_argument("--des_path",type=str)
    
    args = parser.parse_args()
    unzip_file(args.zip_path,args.des_path)
