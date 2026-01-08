
# Video Anomaly Detection using LSTM Autoencoder (PixelPlay Competition)

This repository contains the complete implementation of a **sequence-based Video Anomaly Detection (VAD)** system built for the PixelPlay Kaggle competition. The approach is based on an **LSTM Autoencoder trained only on normal video sequences**, where anomalies are detected using **reconstruction error**.


---

## Overall Pipeline

```
Video
  ↓
Frame Extraction
  ↓
Resize + Normalize
  ↓
Sliding Window Sequences
  ↓
LSTM Autoencoder (train on normal only)
  ↓
Reconstruction Error
  ↓
Temporal Smoothing
  ↓
Frame-wise Anomaly Scores → Submission
```


## Repository Structure

```
pixelplay-vad/
│
├── README.md               ← This file (full explanation)
├── requirements.txt        ← Python dependencies
│         
│
├── src/
│   ├── README.md
│   ├── dataset.py          ← Sequence construction logic
│   ├── feature_extractor.py            
│   ├── model.py            ← LSTM Autoencoder definition
│   ├── train.py            ← Training loop (normal data only)
│   ├── inference.py        ← Reconstruction error scoring
│   └── postprocess.py     ← Smoothing + submission generation
│
└── pixel-play-9.ipynb  ← Executed Kaggle notebook used for submission
```

The `src/` folder contains clean modular code. The notebook is kept only for reproducibility and proof of execution on Kaggle.

---

## Dataset Assumptions

Expected structure:

```
data/
 ├── train/
 │    └── normal/
 │         └── video_id/frame.jpg
 └── test/
      └── video_id/frame.jpg
```

### Why this structure?

* Training is **unsupervised** → only normal videos are used
* Test videos may contain anomalies
* Frame order must be preserved to build sequences

Anomaly detection is based on the assumption:

> The model should only learn what "normal" looks like. Anything it cannot reconstruct well is considered abnormal.

---

## Preprocessing

### 1. Frame Loading

Frames are loaded from disk using OpenCV.

Why frames instead of raw video?

* PyTorch models operate on tensors, not video files
* Extracted frames give direct pixel access
* Allows random access and sliding window generation

---

### 2. Resizing

Frames are resized to **64×64**.

Why resize?

* LSTM input size = H × W
* Memory usage grows quadratically with resolution
* Higher resolution caused GPU memory overflow

Trade-off:

* ✔ Faster training
* ❌ Loss of fine spatial details

---

### 3. Grayscale Conversion
Tried to use Grayscale conversion thought that it would improved the prediction but it failed

Why?

* Motion and structure matter more than color
* Reduces input size by 3×
* Makes training more stable


---

### 4. Normalization

Pixel values are scaled to [0,1].

Why?

* Neural networks train better on small numeric ranges
* Prevents unstable gradients
* Required for MSE reconstruction stability

Feeding raw 0–255 values slows convergence.

---

## Sequence Construction

Frames are grouped into overlapping sequences using sliding windows:

```
[1,2,3,4,5]
[2,3,4,5,6]
[3,4,5,6,7]
...
```

Each training sample = one temporal sequence.

### Why sequences are mandatory

LSTM models temporal dependencies. If we feed only single frames:

* LSTM degenerates into a useless dense layer
* No motion or behavior can be learned

This step converts:

> static image problem → temporal behavior problem

---

## Model Architecture

### LSTM Autoencoder

The model consists of:

1. **Encoder LSTM**

   * Compresses the input sequence into a latent representation

2. **Decoder LSTM**

   * Attempts to reconstruct the original sequence

3. **Linear Output Layer**

   * Maps hidden states back to pixel space

### Why Autoencoder for Anomaly Detection?

Because we want:

* Low error for normal behavior
* High error for unseen abnormal behavior

This avoids the need for frame-level labels.

---

### Why No CNN Feature Extractor?

Ideally, the pipeline should be:

```
Frame → CNN → Feature Vector → LSTM Autoencoder
```

But in this project:

* Only raw pixels were used
* No spatial feature learning

Reason:

* CNN+LSTM pipelines are harder to debug
* Much higher GPU memory usage
* Time constraints during competition

Consequence:

* LSTM had to learn both spatial and temporal patterns
* This severely limited representation quality

---

## Training Strategy

### Normal-Only Training

Only normal video sequences are used during training.

Why?

* If anomalies are included, the model will learn to reconstruct them
* Reconstruction error would no longer signal abnormality

This is a core principle of reconstruction-based anomaly detection.

---

### Loss Function: Mean Squared Error (MSE)

Loss = difference between:

* input sequence
* reconstructed sequence

Why MSE?

* Pixel-wise reconstruction task
* Penalizes small deviations across entire frame

Higher MSE → more abnormal motion pattern.

---

### Optimizer: Adam

Why Adam?

* Stable for RNN training
* Less sensitive to learning rate
* Works well without heavy tuning

SGD would require more careful scheduling and tuning.

---

### Epochs and Batch Size

* Small batch size
* Limited epochs

Why?

* Sequences consume much more memory than images
* Kaggle GPU session limits
* Overfitting observed early

Increasing epochs did not significantly improve validation behavior.

---

## Inference

During testing:

1. Generate sequences
2. Pass through autoencoder
3. Compute reconstruction error

Each sequence receives one anomaly score.

### Mapping Scores Back to Frames

Because submission requires frame-wise scores:

* Sequence scores are aligned to center frames
* Scores are propagated to timeline

This step is required because the model does not predict per-frame directly.

---

##  Post-Processing

### Temporal Smoothing

Rolling average is applied over time.

Why?

* Reconstruction error is noisy
* Slight lighting changes create spikes
* Real anomalies are temporally continuous

Smoothing enforces time consistency.

---

### Video-wise Normalization

Scores are normalized per video.

Why?

* Different videos have different base reconstruction error
* Ranking-based metrics benefit from normalization

This improves relative ordering between frames.

---

## Performance Summary

Final score: approximately **0.58 AUC**

### What Worked

* Proper temporal modeling using LSTM
* Unsupervised normal-only training
* Reconstruction-based anomaly scoring
* Temporal smoothing

### What Limited Performance

* No CNN feature extraction
* Short temporal windows
* High sensitivity to lighting noise
* Weak spatial representations

This was not a tuning problem. It was a representation limitation.

---


## Notebook

The file:

`pixel-play-9.ipynb`

Contains the full Kaggle pipeline:

* Data loading
* Training
* Inference
* Submission

It is already executed to allow reviewers to inspect outputs directly.

---

## Final Note

This repository represents a **learning-focused implementation of temporal anomaly detection** rather than a production-grade or competition-winning system. The design choices were made to balance:

* correctness of anomaly detection principle
* feasibility within Kaggle resource limits
* implementation complexity for a first VAD project

Despite limitations, the project demonstrates a complete end-to-end VAD pipeline using sequence modeling, which is the correct conceptual direction for surveillance anomaly detection.
