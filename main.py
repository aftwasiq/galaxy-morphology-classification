import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)

    def forward(self, x):
        return self.model(x)


def process(model, dLoaders, criterion, optimizer, dSizes, num_epochs=25, device=None):
    model.to(device)
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            losses = 0.0

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

            epoch_loss = (losses / dSizes[phase])
            print(f"Epoch #: {epoch}/{num_epochs - 1}, {phase} Loss: {epoch_loss:.4f}")

    return model

