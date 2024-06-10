import numpy as np
from PIL import ImageOps
import matplotlib.pyplot as plt

def transposeShow(image):
	return np.transpose(image.squeeze(), axes=(1,2,0))

def printImg(dataTrainL, classes):
	for image, id in dataTrainL:
		print(classes)
		batch_size = image.shape[0]
		if batch_size > 4:
			ncols = 4
		else:
			ncols = batch_size
		nrows = (batch_size + ncols - 1) // ncols
		fig, axes = plt.subplots(nrows, ncols, figsize=(19, 10))
		axes = axes.flatten()
		for i in range(batch_size):
			ax = axes[i]
			ax.imshow(transposeShow(image[i]))
			ax.set_title(f'ID: {classes[id[i].item()]}')
			ax.axis('off')
		break
	plt.show()

def pad_to_square(image):
    width, height = image.size
    max_dim = max(width, height)
    padding = (max_dim - width, max_dim - height)
    padding = (padding[0] // 2, padding[1] // 2)
    return ImageOps.expand(image, padding, fill=(255, 255, 255))
