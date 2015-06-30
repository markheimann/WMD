#start
import pdb, sys, numpy as np, pickle, multiprocessing as mp
from scipy.stats import mode
#OPTIONAL: implement full WMD
#TODO: figure out why accuracy varies so dramatically with different train/test splits
#TODO: figure out why rwmd is a bit slow (see note in rwmd method) 
#TODO: possibly try a simpler representation and sklearn kNN for comparison (see if it's the data or this method)

def distance(x1,x2):
    return np.sqrt( np.sum((np.array(x1) - np.array(x2))**2) )

#Get word centroid of each document (represent each document as a weighted average of its word vectors)
def getWordCentroid(document, nbow):
	#basically sum columns multiplied by corresponding entries in normalized bag of words
	return np.dot(document, nbow)

#Compute the Word Centroid Distance
def wcd(doc1, doc2, doc1_nbow, doc2_nbow):
	return distance(getWordCentroid(doc1, doc1_nbow), getWordCentroid(doc2, doc2_nbow))

#Given a word and another document, find the word's nearest neighbor word vector in the other document
def find_wordNN(word, document):
	#Keep track of nearest word (by its index in the document) and its distance to the word in question
	nearest_neighbor = None
	nearest_distance = float("inf")
	#Search through all words in the other document
	for neighbor_index in range(0,document.shape[1]): #words are in columns so iterate over columns
		neighbor_wordVec = document[:,neighbor_index]
		neighbor_dist = distance(word, neighbor_wordVec)
		#Update nearest neighbor and distance if a nearer neighbor is found
		if neighbor_dist < nearest_distance:
			nearest_neighbor = neighbor_index
			nearest_distance = neighbor_dist
	return (nearest_neighbor, nearest_distance) 

#Solution to WMD with 1 constraint (see Kusner et. al, 2015)
#For each word in document 1, find its nearest neighbor in document 2
#Transport all its weight (as given by its normalized bag-of-words entry) to that nearest neighbor
def lessConstrained_docDistance(doc1, doc2, doc1_nbow):
	distance = 0
	for word_index in range(0,doc1.shape[1]): 
		word = doc1[:,word_index]
		nearest_word, nearest_distance = find_wordNN(word, doc2)
		distance += nearest_distance * doc1_nbow[word_index]
	return distance

#Perform nearest neighbor search in both directions
#(find nearest neighbors in document 2 of words in document 1 and vice versa)
#take the maximum for a pretty tight bound on exact WMD, but faster (Kusner et. al, 2015)
def rWMD(doc1, doc2, doc1_nbow, doc2_nbow):
	#NOTE: this definitely duplicates some pairwise word vector distance calculations and can/should be made more efficient
	return max(lessConstrained_docDistance(doc1, doc2, doc1_nbow), lessConstrained_docDistance(doc2, doc1, doc2_nbow))

#Given a query document and a training set, return k nearest neighbors
def kNN(query_doc, query_nbow, train_data, train_nbow, num_neighbors):
	#Keep track of nearest neighbors and their distances (initially we have none)
	nn_distances = [float("inf")] * num_neighbors #keep ordered list of k nearest neighbors, largest to smallest distance
	nearest_neighbors = [None] * num_neighbors #keep track of indices of nearest neighbors

	#Documents are entries in a list of training data (each document is a matrix whose columns are word vectors)
	for train_doc_index in xrange(0,train_data.size):
		train_doc = train_data[train_doc_index]
		train_doc_nbow = train_nbow[train_doc_index]
		dist = wcd(query_doc, train_doc, query_nbow, train_doc_nbow)
		#dist = rWMD(query_doc, train_doc, query_nbow, train_doc_nbow)
		nn_count = 0 #we'll be iterating over the list of nearest neighbors we have so far
		while dist < nn_distances[nn_count]: #see if this distance is shorter than the distance to a nearest neighbor
			nn_count += 1
			if nn_count >= num_neighbors: #we've gone through all the nearest neighbors
				break
		if nn_count > 0: #this is one of the k nearest neighbors so far
			nn_distances.insert(nn_count, dist)
			nearest_neighbors.insert(nn_count, train_doc_index) #keep nn list in sync with nn distance list
			nearest_neighbors.pop(0) #remove the furthest-away NN (as it's no longer among the closest known k)
			nn_distances.pop(0)
	return nearest_neighbors

#include labels if you want to test performance
def testWithLabels(num_neighbors, train_data, test_data, train_nbow, test_nbow, train_labels, test_labels):
	correct_preds = 0
	num_preds = 0
	num_test = 100
	num_test = test_data.size
	for query_index in xrange(0,num_test):
		query_doc = test_data[query_index]
		query_nbow = test_nbow[query_index]
		query_NN = kNN(query_doc,query_nbow,train_data,train_nbow,num_neighbors)
		NN_labels = train_labels[query_NN]
		predicted_label = mode(NN_labels)[0][0] #get modal value
		if predicted_label == test_labels[query_index]:
			correct_preds += 1
		num_preds += 1
		if True: #num_preds % 10 == 0: #print out every 10th
			print("Made %d of %d predictions" % (num_preds, num_test)),
			if num_preds % 100 == 0: #print accuracy every 100th time
				print( "with accuracy %f" % (float(correct_preds) / num_preds))
			else:
				print
	accuracy = float(correct_preds) / num_test
	return accuracy


if __name__ == "__main__":
	#User arguments: train_data_file test_data_file num_neighbors  <optional train labels> <optional test labels>
	#If user provides train and test labels then they want to compute accuracy on a set of documents
	#If user doesn't provide train and test labels then they want to get the nearest neighbors of a single query
	train_data_file = sys.argv[1]
	test_data_file = sys.argv[2]
	num_neighbors = int(sys.argv[3])

	#Load in train data now
	with open(train_data_file,"r") as tdf:
		[train_data, train_nbow, CTr, wordsTr] = pickle.load(tdf)
	if len(sys.argv) == 6: #user has provided labels
		train_labels_file = sys.argv[4]
		test_labels_file = sys.argv[5]
		#Load in train and test labels
		with open(train_labels_file,"r") as trlf:
			train_labels = pickle.load(trlf)
		with open(test_labels_file, "r") as telf:
			test_labels = pickle.load(telf)	
		#since we have labels, we know that test data is a collection of documents
		with open(test_data_file, "r") as tedf:
			[test_data, test_nbow, CTe, wordsTe] = pickle.load(tedf)
		test_accuracy = testWithLabels(num_neighbors, train_data, test_data, train_nbow, test_nbow, train_labels, test_labels)   
		print "Test accuracy: ", test_accuracy

	elif len(sys.argv) == 4: #user hasn't provided labels
		#so now we know test data is a single document
		with open(test_data_file, "r") as tedf:
			[test_data, test_nbow, CTe, wordsTe] = pickle.load(tedf)
		recommendations = kNN(test_data, test_nbow, train_data, train_nbow, num_neighbors)
		print "We recommend the following documents: ", recommendations

	else: #rudimentary input error handling
		print "Usage: python approx_wmd.py [train_data_vectors] [test_data_vectors] [num_neighbors]"
		print "Optional: add [train_data_labels] [test_data_labels] for testing at the end"
