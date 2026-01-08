import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FrameDataset(Dataset):
    """
    Loads individual frames and keeps video-wise ordering.
    CNN features are extracted later.
    """

    def __init__(self, root_dir, img_size=224):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        for vid in sorted(os.listdir(root_dir)):
            vpath = os.path.join(root_dir, vid)
            if not os.path.isdir(vpath):
                continue
            for f in sorted(os.listdir(vpath)):
                if f.endswith(".jpg"):
                    self.samples.append((vid, os.path.join(vpath, f)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return vid, img
