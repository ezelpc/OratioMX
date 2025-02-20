import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class signLanguageModel:
    def __init__(self, model_path = "modelo_senas_mobilenet.pth", num_classes = 50):
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        self.model.load_state_dict(torch.load('model_path', map_location=device))
        self.model.to(device).eval()

        self.transform = transforms.Compose([
            transforms.Resize(224,224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(image)
            predicted_class = torch.argmax(output).item()
        
        return predicted_class
    
            
        