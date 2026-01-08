import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoSequenceDataset(Dataset):
    """
    Creates sliding window sequences from video frames.
    Each sample = (T, H, W) flattened before LSTM.
    """

    def __init__(self, root_dir, seq_len=10, img_size=64, train=True):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.img_size = img_size
        self.train = train

        self.samples = []  # list of (video_dir, start_idx)

        for vid in sorted(os.listdir(root_dir)):
            vpath = os.path.join(root_dir, vid)
            frames = sorted(os.listdir(vpath))
            for i in range(len(frames) - seq_len):
                self.samples.append((vpath, i))

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        return img

    def __getitem__(self, idx):
        vpath, start = self.samples[idx]
        frames = sorted(os.listdir(vpath))[start:start + self.seq_len]

        seq = []
        for f in frames:
            fp = os.path.join(vpath, f)
            img = self._load_frame(fp)
            seq.append(img)

        seq = np.stack(seq)  # (T, H, W)
        seq = seq.reshape(self.seq_len, -1)  # flatten for LSTM

        return torch.tensor(seq, dtype=torch.float32)
