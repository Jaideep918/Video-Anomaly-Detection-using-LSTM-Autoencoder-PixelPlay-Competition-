import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import FrameDataset
from feature_extractor import FeatureExtractor
from model import LSTMAutoEncoder

SEQ_LEN = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer(root_dir):
    ds = FrameDataset(root_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    cnn = FeatureExtractor().to(DEVICE).eval()
    lstm = LSTMAutoEncoder(512).to(DEVICE)
    lstm.load_state_dict(torch.load("lstm_ae.pth", map_location=DEVICE))
    lstm.eval()

    buffer = []
    scores = []
    ids = []

    with torch.no_grad():
        for vid, img in loader:
            img = img.to(DEVICE)
            feat = cnn(img)[0]

            buffer.append(feat)
            ids.append(vid[0])

            if len(buffer) == SEQ_LEN:
                seq = torch.stack(buffer).unsqueeze(0)
                recon = lstm(seq)
                err = torch.mean((recon - seq) ** 2).item()

                scores.append(err)
                buffer.pop(0)

    np.save("scores.npy", np.array(scores))
    np.save("vids.npy", np.array(ids[SEQ_LEN-1:]))


if __name__ == "__main__":
    infer("data/test")
