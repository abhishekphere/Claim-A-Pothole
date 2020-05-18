
import torch

from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

from flask import Flask, request

app = Flask(__name__)
app.debug = True

@app.route('/', methods=['GET', 'POST'])
def handle_request():
    return "Connected to Server successfully"

@app.route('/predict', methods=["POST"])
def isPothole():
    img = Image.open(request.files['image']).convert('RGB')
    return str(prediction(img))

def prediction(img):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = torch.load("Your saved model")
    model.eval()

    return helper(img, model, device, test_transforms)

def helper(image, model, device, test_transforms):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()

    return index

if __name__ == '__main__':
    # app.run(host='192.168.1.59', port=5000, debug=True)
    app.run(debug=True)