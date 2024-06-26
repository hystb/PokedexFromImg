import torchvision as tv
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from utils import *
from CNN import CNN

def train(model=None, learningRate=0.001, nEpoch=50, pathToData='./PokemonData', device='cpu', batchSize='1'):
	preprocessData = tv.transforms.Compose(
	    [tv.transforms.Lambda(lambda image: image.convert('RGB')),
	     tv.transforms.Lambda(pad_to_square),
	     tv.transforms.Resize((128, 128)),
	     tv.transforms.ToTensor()]
	    )
	dataTrain = tv.datasets.ImageFolder(root=pathToData, transform=preprocessData)
	dataTrainL = torch.utils.data.DataLoader(dataTrain,
	                                         batch_size = batchSize,
	                                         shuffle=False)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	classes_path = f"classes_b{timestamp}.pkl"
	with open(classes_path, 'wb') as f: pickle.dump(dataTrain.classes, f)
	if model == None:
		model = CNN(len(dataTrain.classes))
	model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learningRate)
	losses = np.zeros(nEpoch)

	for epoch in range(nEpoch):
		actualLoss = 0.0
		for image, id in dataTrainL:
			image, id = image.to(device), id.to(device)
			optimizer.zero_grad()
			answer = model.forward(image)
			loss = criterion(answer, id)
			loss.backward()
			optimizer.step()
			actualLoss += loss.item()
		print(f"epoch: {epoch} actual, losses {actualLoss}")
		losses[epoch] = actualLoss
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	pathToSave = f"model_{timestamp}.pth"
	torch.save(model.state_dict(), pathToSave)
	classes_path = f"classes_a{timestamp}.pkl"
	with open(classes_path, 'wb') as f: pickle.dump(dataTrain.classes, f)
	print(f"End of training\nModel parameters saved in {pathToSave}\nDataTrain.classes saved in {classes_path}")

