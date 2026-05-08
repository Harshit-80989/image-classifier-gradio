import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
from urllib.request import urlopen

# Load pretrained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Load ImageNet class labels
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes = urlopen(labels_url).read().decode("utf-8").splitlines()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Prediction function
def predict(image):
    image = image.convert("RGB")
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    top3 = torch.topk(probs, 3)

    results = {}
    for i in range(3):
        results[classes[top3.indices[i]]] = float(top3.values[i]) * 100

    return results

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🧠 Image Classifier",
    description="Upload an image and get top predictions using ResNet18"
)

demo.launch()
