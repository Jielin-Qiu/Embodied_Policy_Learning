from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import torch


class ImageCaptionDataset(Dataset):
    def __init__(self, caption, exec, img_paths, resolution, mean, std, device, init_graphs, embed_model = None, tokenizer=None, max_len=None, id = None):
        self.caption = caption
        self.img_paths = img_paths
        self.tokenizer = tokenizer
        self.max_len = max_len 
        self.resolution = resolution
        self.mean = mean
        self.std = std
        self.id = id
        self.exec = exec
        self.embed_model = embed_model
        self.device = device
        self.init_graphs = init_graphs

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        caption = self.caption[idx]
        id = self.id[idx]
        init_graph = self.init_graphs[idx]
        exec = self.exec[idx].lower()

        img = Image.open(img_path)
        
        transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(), 
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        exec_inputs = self.tokenizer([exec], max_length = self.max_len, padding = 'max_length', truncation = True, return_tensors='pt').to(self.device)
        exec_inputs = exec_inputs.input_ids
        patch_img = transform(img)

        return caption, patch_img, id, exec_inputs, exec, init_graph
