import json
import numpy as np
import torch
from PIL import Image
from timm.models import create_model
from torchvision import transforms


class ModelPredictor(object):
    def __init__(self, args):
        self.model = create_model(
            args.model,
            pretrained = False,
            num_classes = args.nb_classes,
        )
        self.tfsm = transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0,0,0], std = [1,1,1])])
        
        self.device = args.device
        self.model.load_state_dict(torch.load(args.resume, map_location=self.device)["model"])
        self.model.to(self.device)
        self.model.eval()

        with open("./dataset/name_mapping.json", 'r') as INFILE:
            self.mapping = json.load(INFILE)

    def preprocess(self, input):
        output = self.tfsm(input).to(self.device)
        output = output.unsqueeze(0)
        return output
    
    def predict(self, image):
        image = self.preprocess(image)
        with torch.no_grad():
            output = self.model(image)
            output = output.softmax(1).to("cpu").numpy()

        output = np.argmax(output)
        medical_leaf_name = self.mapping[str(output)]

        return {"medical_leaf_name" : medical_leaf_name}