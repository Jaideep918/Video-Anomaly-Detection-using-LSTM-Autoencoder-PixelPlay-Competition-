import numpy as np
import pandas as pd


def smooth(x, k=7):
    return np.convolve(x, np.ones(k)/k, mode="same")


scores = np.load("scores.npy")
vids = np.load("vids.npy")

df = pd.DataFrame({"vid": vids, "Predicted": scores})

df["Predicted"] = df.groupby("vid")["Predicted"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
)

df["Id"] = df["vid"] + "_" + df.groupby("vid").cumcount().astype(str)
df = df[["Id", "Predicted"]]

df.to_csv("submission.csv", index=False)
print("Saved submission.csv")
