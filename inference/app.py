import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image

# Define the CNN (same as training)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Initialize Flask app
app = Flask(__name__)

# Define preprocessing transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Service is running"}), 200

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
        model = Net()
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
