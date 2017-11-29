import math
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
# import sklearn


def get_data():
	path = 'data/'
	red_wine = pd.read_csv(path+'winequality-red.csv', sep=';')
	white_wine = pd.read_csv(path+'winequality-white.csv', sep=';')
	# 1599 entries in total
	# print red_wine.describe()
	return red_wine, white_wine

def check_for_duplicates(red_wine, white_wine):
	number_of_red_duplicates = np.sum(np.array(red_wine.duplicated()))
	number_of_white_duplicates = np.sum(np.array(white_wine.duplicated()))
	print 'Number of red duplicates : ', number_of_red_duplicates
	print 'Number of white duplicates : ', number_of_white_duplicates
	return number_of_red_duplicates, number_of_white_duplicates


def get_variation(count, colour):
	plt.xlabel("Quality")
	plt.ylabel("Samples")
	plt.title('Figure 1: quality distribution of red wine')
	count.plot(kind='bar',color=colour)
	plt.show()

# def split_data(_data):
# 	# 70/30 split
# 	X_train, X_test, y_train, y_test = train_test_split(_data.data, _data.target, test_size=0.3, random_state=0)



def main():
	red_wine, white_wine = get_data()
	number_of_red_duplicates, number_of_white_duplicates = check_for_duplicates(red_wine, white_wine)

	red_counts=red_wine.groupby('quality').size()
	white_counts = white_wine.groupby('quality').size()

	get_variation(red_counts, 'r')
	get_variation(white_counts, 'g')


	## stub ##


if __name__ == '__main__':
	main()