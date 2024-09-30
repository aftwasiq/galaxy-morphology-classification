import torch
# import pandas as pd
# import time as
# import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from main import classifier
import torch.optim as optim
from flask import Flask, request, render_template, jsonify
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 30

meanv = [
    0.485,  # red
    0.456,  # green
    0.406  # blue
]

standard = [
    0.229,  # r
    0.224,  # g
    0.225  # b
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[meanv[0], meanv[1], meanv[2]], std=[standard[0], standard[1], standard[2]])
])

tData = datasets.ImageFolder("dataset/train", transform=transform)
vData = datasets.ImageFolder("dataset/val", transform=transform)
# tLoader = torch.utils.data.DataLoader(tData, batch_size=12, shuffle=False)
# vLoader = torch.utils.data.DataLoader(vData, batch_size=12, shuffle=True)
tLoader = torch.utils.data.DataLoader(tData, batch_size=14, shuffle=True)
vLoader = torch.utils.data.DataLoader(vData, batch_size=14, shuffle=False)
dSize = {"train": len(tData), "val": len(vData)}
dLoader = {"train": tLoader, "val": vLoader}

model = classifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()


def proccess(model, dLoaders, criterion, optimizer, dSizes, num_epochs=30, device=None):
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            losses, correct = 0.0, 0.0

            if phase == "train":
                model.train()
            elif phase == "val":
                model.eval()
            else:
                raise ValueError("could not process set")

            for inputs, labels in dLoaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                losses += (loss.item() * inputs.size(0))
                _, predictions = torch.max(outputs, 1)
                correct += torch.sum(torch.eq(predictions, labels.data)).item()

            epoch_loss = (losses / dSizes[phase])
            epoch_acc = (correct / dSizes[phase])
            print(f"Epoch #: {epoch}/{num_epochs - 1}, Phase: {phase}, " +
                  f"Loss: {epoch_loss:.4f} Accuracy (dec): {epoch_acc:.4f}")

    return model


processed_model = proccess(model, dLoader, criterion, optimizer, dSize, num_epochs=epochs, device=device)
torch.save(processed_model.state_dict(), "model.pth")

app = Flask(__name__)
model = classifier().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    class_labels = ["Elliptical", "Irregular", "Spiral"]
    prediction = class_labels[predicted.item()]

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
