from scipy.stats import ttest_ind
import numpy as np

#Read in data from file: each float is on its own line
#Return numpy array with data
def readFloats(fileName):
	data = []
	data_file = open(fileName, "r")
	for line in data_file:
		data.append( float(line.strip()) )
	return np.asarray(data)

#Want to know difference in means: use a t-test and return p-value
def t_test(dataset1, dataset2):
	return ttest_ind(dataset1, dataset2)

if __name__ == "__main__":
	bow_accuracy = readFloats("bow_knnAccuracy.txt")
	wmd_accuracy = readFloats("wmd_knnAccuracy.txt")
	bow_meanAccuracy = np.mean(bow_accuracy)
	wmd_meanAccuracy = np.mean(wmd_accuracy)
	test_stats = t_test(bow_accuracy, wmd_accuracy)
	print "Mean accuracy when using BOW representation: ", bow_meanAccuracy
	print "Mean accuracy when using WMD representation: ", wmd_meanAccuracy
	print "Difference in accuracy: ", wmd_meanAccuracy - bow_meanAccuracy
	print
	print "Null hypothesis that there's no difference between mean BOW and WMD accuracy. ",
	print("t-statistic: %f.  p-value: %f" % (test_stats[0], test_stats[1]))
