import pathlib

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


CLASSES = sorted([
    "Anaheim", "Bakersfield", "Los_Angeles",
    "Riverside", "SLO", "San_Diego",
])


class FineTunedModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def predict(test_dir):
    test_dir = pathlib.Path(test_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = pathlib.Path(__file__).resolve().parent

    model = FineTunedModel(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(script_dir / "model.pt", map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    predictions = {}
    with torch.no_grad():
        for path in sorted(test_dir.glob("*.jpg")):
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            output = model(tensor)
            predicted_index = output.argmax(dim=1).item()
            predictions[path.name] = CLASSES[predicted_index]

    return predictions


if __name__ == "__main__":
    preds = predict("./data")
    correct = 0
    total = 0
    for filename, pred in preds.items():
        true_label = filename.rsplit("-", 1)[0]
        if pred == true_label:
            correct += 1
        total += 1
    print(f"Accuracy on training data: {correct}/{total} = {correct/total:.4f}")
