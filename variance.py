# Total number of white wine instance = 4898
# Number of duplicates in white_wine = 937
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_data():
	path = 'data/'
	red_wine = pd.read_csv(path+'winequality-red.csv', sep=';')
	white_wine = pd.read_csv(path+'winequality-white.csv', sep=';')
	# 1599 entries in total
	# print red_wine.describe()
	print "Length of white wine : ", len(white_wine)
	print "Length of red wine : ", len(red_wine)
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
	plt.title('Figure 1: quality distribution of white wine')
	count.plot(kind='bar',color=colour)
	plt.show()


def main():
	white_wine = get_data()
	# sns.pairplot(pd.DataFrame(white_wine))
	# f, ax = pl.subplots(figsize=(10, 8))
	# plt.matshow(white_wine)
	number_of_white_duplicates = check_for_duplicates(white_wine)
	white_counts = white_wine.groupby('quality').size()
	get_variation(white_counts, 'b')


if __name__ == '__main__':
	main()