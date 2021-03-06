import numpy as np
import pandas as pd
from utils import get_data
import matplotlib.pyplot as plt



label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def show_image(pixel_data):
	pass


def main():
	
	X, Y = get_data(balance_class=False)

	while True:
		for i in range(len(label_map)):
			x, y = X[Y == i], Y[Y == i]
			N = len(y)
			j = (np.random.choice(N))
			plt.imshow(x[j].reshape(48, 48), cmap='gray')
			plt.title(label_map[int(y[j])])
			plt.show()
		prompt = input('Quit? Enter Y:\n')
		if prompt == 'Y':
			break



if __name__ == '__main__':
	main()