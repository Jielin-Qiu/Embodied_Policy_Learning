import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import schema
from PIL import Image
from torchvision import transforms

class ImageCaptionDataset(Dataset):
    def __init__(self, caption, img_paths, tokenizer, max_len):
        self.caption = caption
        self.img_paths = img_paths
        self.tokenizer = tokenizer
        self.max_len = max_len 

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        caption = self.caption[idx]

        img = Image.open(img_path)
        
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 256
        transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ])

        inputs = self.tokenizer([caption], max_length = self.max_len, padding = 'max_length', truncation = True, return_tensors='pt').input_ids
        patch_img = transform(img)

        return inputs, patch_img


def get_loader(train_tsv, val_tsv, test_tsv, tokenizer):
    train_img_paths = train_tsv[:, 1]
    train_captions = train_tsv[:, 2]

    val_img_paths = val_tsv[:, 1]
    val_captions = val_tsv[:, 2]

    test_img_paths = test_tsv[:, 1]
    test_captions = test_tsv[:, 2]

    train_ds = ImageCaptionDataset(
        caption = train_captions,
        img_paths = train_img_paths,
        tokenizer = tokenizer,
        max_len = schema.max_len
    )
    val_ds = ImageCaptionDataset(
        caption = val_captions,
        img_paths = val_img_paths,
        tokenizer = tokenizer,
        max_len = schema.max_len
    )
    test_ds = ImageCaptionDataset(
        caption = test_captions,
        img_paths = test_img_paths,
        tokenizer = tokenizer,
        max_len = schema.max_len
    )

    train_loader = DataLoader(
        dataset = train_ds,
        batch_size = schema.batch_size,
        shuffle = True
    )
    val_loader = DataLoader(
        dataset = val_ds,
        batch_size = schema.batch_size,
        shuffle = True
    )
    test_loader = DataLoader(
        dataset = test_ds,
        batch_size = schema.batch_size,
        shuffle = True
    )

    return train_loader, val_loader, test_loader

    





