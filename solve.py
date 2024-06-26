from utils import *
from CNN import CNN
import pickle
import torchvision as tv
import torch
from main import loadModel

def solve(classesPath, modelPath, device, pathToTest, batchSize=64):
    with open(classesPath, 'rb') as f:
        classes = pickle.load(f)
        print(len(classes))
        model = CNN(len(classes))
    model = loadModel(model, modelPath, device)
    print("Model Loaded")
    preprocessData = tv.transforms.Compose(
	    [tv.transforms.Lambda(lambda image: image.convert('RGB')),
	     tv.transforms.Lambda(pad_to_square),
	     tv.transforms.Resize((128, 128)),
	     tv.transforms.ToTensor()]
	    )
    dataTrain = tv.datasets.ImageFolder(root=pathToTest, transform=preprocessData)
    dataTrainL = torch.utils.data.DataLoader(dataTrain,
	                                         batch_size = batchSize,
	                                         shuffle=False)
    testImg(dataTrainL, classes, model, device, batchSize)
    
    
        
    
