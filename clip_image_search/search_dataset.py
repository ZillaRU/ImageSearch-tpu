from torch.utils.data import Dataset
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClipSearchDataset(Dataset):
    def __init__(self, img_dir,  img_ext_list = ['.jpg', '.png', '.jpeg', '.tiff'], preprocess = None, mode = None, updeate_list = None):
        self.preprocess = preprocess
        self.img_path_list = []
        self.update_list = updeate_list
        if mode is None:
            self.walk_dir(img_dir, img_ext_list)
            print(f'Found {len(self.img_path_list)} images in {img_dir}')
        elif mode == 'update' and updeate_list is not None:
            self.update_db()
            print(f'Found {len(self.img_path_list)} new images in {img_dir}')


    def walk_dir(self, dir_path, img_ext_list): # work for symbolic link
        for root, dirs, files in os.walk(dir_path):
            self.img_path_list.extend(
                os.path.join(root, file) for file in files 
                if os.path.splitext(file)[1].lower() in img_ext_list
            )
            for dir in dirs:
                full_dir_path = os.path.join(root, dir)
                if os.path.islink(full_dir_path):
                    self.walk_dir(full_dir_path, img_ext_list)

    def update_db(self):
        for img_path in self.update_list:
            self.img_path_list.append(img_path) # root+dir+files



    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img)
        return img, img_path
    

