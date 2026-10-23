import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from google.colab import files

MODEL_PATH = "busi_model.pt"  
CLASS_NAMES = ["benign", "malignant", "normal"]   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BreastUltrasoundModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.base_model = models.efficientnet_b0(weights=None) 
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)


model = BreastUltrasoundModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()   


test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

uploaded = files.upload()
IMAGE_PATH = list(uploaded.keys())[0]

img = Image.open(IMAGE_PATH).convert("RGB")
x = test_tf(img).unsqueeze(0).to(device)  

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

pred = int(probs.argmax())
print(f"Prediction: {CLASS_NAMES[pred]} ({probs[pred]*100:.1f}%)")
for name, p in zip(CLASS_NAMES, probs):
    print(f"  {name:10s} {p*100:5.1f}%")


plt.imshow(img, cmap="gray")
plt.title(
    f"There is a {probs[pred] * 100:.1f}% model-estimated probability "
    f"that this image belongs to the {CLASS_NAMES[pred]} class.\n",
    fontsize=10
)
plt.axis("off")
plt.show()

print(f"There are {probs[pred]*100:.1f}% chances that it would be {CLASS_NAMES[pred]} cancer cell. Please consult a doctor for actual diagnosis.")