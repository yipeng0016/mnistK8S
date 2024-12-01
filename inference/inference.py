import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image

# Define the CNN (same as training)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Initialize Flask app
app = Flask(__name__)

# Define preprocessing transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    try:
        # Load and preprocess the image
        image = Image.open(file).convert("L")  # Convert to grayscale
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Define the device as CPU
        device = torch.device("cpu")

        # Load model
        model = SimpleCNN()
        model.load_state_dict(torch.load("/mnt/cnn_model.pth"))
        model.to(device)
        model.eval()

        # Perform prediction
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        prediction = output.argmax().item()

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
