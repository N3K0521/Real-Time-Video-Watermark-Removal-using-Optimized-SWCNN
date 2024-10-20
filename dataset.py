import os
import glob
import random
import torch
import torch.utils.data as udata
import torchvision.transforms as transforms
from PIL import Image
def normalize(data):
    return data / 255.


class Dataset(udata.Dataset):
    def __init__(self, train=True, mode='gray', data_path='/media/npu/Data/jtc/data/', patch_size=256, stride=128, aug_times=1):
        super(Dataset, self).__init__()
        self.train = train
        self.mode = mode
        self.data_path = data_path
        self.patch_size = patch_size
        self.stride = stride
        self.aug_times = aug_times

        # Define torchvision transforms for data augmentation and normalization
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(patch_size),  # Random crop and resize
                transforms.RandomRotation(90),  # Random rotation
                transforms.ToTensor(),  # Convert image to tensor
                # transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale image to [-1, 1]
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.RandomResizedCrop(patch_size),  # Random crop and resize
                # transforms.RandomRotation(90),  # Random rotation
                transforms.ToTensor(),  # Convert image to tensor
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB image to [-1, 1]
            ])

        if self.train:
            self.data_files = self.load_train_data()
        else:
            self.data_files = self.load_validation_data()

    def load_train_data(self):
        """Load all training data file paths"""
        if self.mode == 'gray':
            files = glob.glob(os.path.join(self.data_path, 'train', '*.png'))
        elif self.mode == 'color':
            files = glob.glob(os.path.join(self.data_path, 'train', '*.jpg'))

        files.sort()
        return files

    def load_validation_data(self):
        """Load all validation data file paths"""
        if self.mode == 'gray':
            files = glob.glob(os.path.join(self.data_path, 'Set12', '*.png'))
        elif self.mode == 'color':
            files = glob.glob(os.path.join(self.data_path, 'SWCNN_test', '*.jpg'))

        files.sort()
        return files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        """Reads an image, applies transformations"""
        img_path = self.data_files[index]
        img = Image.open(img_path)  # Read image using PIL

        # Apply transformations
        img = self.transform(img)


        return img


if __name__ == '__main__':
    # Example usage:
    data_path = '/root/autodl-tmp/data'

    # Create train dataset
    train_dataset = Dataset(train=True, mode='color', data_path=data_path, patch_size=256, stride=128, aug_times=1)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=4)
    for i, data in enumerate(train_loader, 0):
        print(data.shape)
    print(f"Number of training images: {len(train_dataset)}")

    # Create validation dataset
    val_dataset = Dataset(train=False, mode='color', data_path=data_path)
    print(f"Number of validation images: {len(val_dataset)}")