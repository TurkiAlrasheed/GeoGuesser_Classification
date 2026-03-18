# SoCalGuessr

A GeoGuessr-style web game where you compete head-to-head against an AI to identify Southern California cities from street-view images.

You'll see a photo each round and pick from 6 cities: **Anaheim**, **Bakersfield**, **Los Angeles**, **Riverside**, **San Luis Obispo**, and **San Diego**. The AI makes its own prediction — after 10 rounds, see who wins.

## Running the Game

### 1. Clone the repo

```bash
git clone https://github.com/TurkiAlrasheed/GeoGuesser_Classification.git
cd GeoGuesser_Classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the game

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## Project Structure

```
├── app.py              # Flask web app (game server)
├── train_vit.py        # Model training script
├── predict_vit.py      # Standalone inference script
├── model.pt            # Trained model weights
├── history.json        # Training history (loss, accuracy)
├── training_curve.png  # Training visualization
├── requirements.txt    # Python dependencies
├── data/               # Street-view images
├── templates/          # HTML templates
└── static/             # CSS and JavaScript
```

---

## About the Model

The AI opponent is an EfficientNet-B0 fine-tuned on ~9,000 street-view images to classify which of the 6 cities an image belongs to.

### Architecture

- EfficientNet-B0 pretrained on ImageNet
- Final classification layer replaced for 6 output classes
- Early feature layers frozen during fine-tuning

### Training

- Optimizer: AdamW (lr: 1e-4, weight decay: 1e-4)
- Scheduler: cosine annealing
- Loss: cross-entropy
- Batch size: 32
- Epochs: 15

### Data Processing

All images resized to 224x224. Training augmentations: random horizontal flip, rotation, color jitter, and affine translation. Validation images were only resized and normalized.

### Results

- **92.7%** validation accuracy
- **95.5%** training accuracy

### Retraining

To retrain from scratch (optional — a trained `model.pt` is already included):

```bash
python train_vit.py
```
