import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_data():
	path = 'data/'
	red_wine = pd.read_csv(path+'winequality-red.csv', sep=';')
	white_wine = pd.read_csv(path+'winequality-white.csv', sep=';')
	# 1599 entries in total
	# print red_wine.describe()
	print "Length of white wine : ", len(white_wine)
	return white_wine

def check_for_duplicates(, white_wine):
	number_of_white_duplicates = np.sum(np.array(white_wine.duplicated()))
	print 'Number of white duplicates : ', number_of_white_duplicates
	return number_of_white_duplicates


def get_variation(count, colour):
	plt.xlabel("Quality")
	plt.ylabel("Samples")
	plt.title('Figure 1: quality distribution of white wine')
	count.plot(kind='bar',color=colour)
	plt.show()


def main():
	white_wine = get_data()
	number_of_white_duplicates = check_for_duplicates(white_wine)
	# get the range of values in each of the coulms
	
	# white_counts = white_wine.groupby('quality').size()
	# get_variation(white_counts, 'b')


if __name__ == '__main__':
	main()