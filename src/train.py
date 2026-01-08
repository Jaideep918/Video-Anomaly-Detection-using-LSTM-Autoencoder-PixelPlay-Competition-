import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FrameDataset
from feature_extractor import FeatureExtractor
from model import LSTMAutoEncoder

SEQ_LEN = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(root_dir):
    ds = FrameDataset(root_dir)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    cnn = FeatureExtractor().to(DEVICE).eval()
    lstm = LSTMAutoEncoder(512).to(DEVICE)

    opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    buffer = []
    lstm.train()

    for epoch in range(5):
        total = 0
        for vid, imgs in loader:
            imgs = imgs.to(DEVICE)
            feats = cnn(imgs)

            for f in feats:
                buffer.append(f)
                if len(buffer) == SEQ_LEN:
                    seq = torch.stack(buffer).unsqueeze(0)
                    recon = lstm(seq)
                    loss = loss_fn(recon, seq)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    total += loss.item()
                    buffer.pop(0)

        print("Epoch", epoch, "Loss", total)

    torch.save(lstm.state_dict(), "lstm_ae.pth")


if __name__ == "__main__":
    train("data/train/normal")
