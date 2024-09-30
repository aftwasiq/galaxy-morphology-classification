'''from flask import Flask, render_template, request, redirect, url_for
import torch
import os
from torchvision import transforms
from PIL import Image
import os
from main import classifier, process
app = Flask(__name__)

model = classifier()
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        image_path = os.path.join("static", file.filename)
        file.save(image_path)

        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        class_names = {0: "Elliptical", 1: "Spiral", 2: "Irregular"}
        predicted_class = class_names.get(predicted.item(), "Unknown")

        return render_template("index.html", prediction=predicted_class, image_file=file.filename)


if __name__ == "__main__":
    app.run(debug=True)'''
