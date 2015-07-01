import pdb, sys, numpy as np, pickle 
sys.path.append('python-emd-master') #for importing specialized emd (earth mover's distance) solver
from scipy.stats import mode
from emd import emd #for computing wmd distance using specialized emd solver
import math, random
#NOTE: emd solver does not work on OS X.  It does, however, work on Ubuntu. 
#If on OS X, comment out emd import and usage and do not use WMD (use heuristic such as rWMD)
#(or just run it on a virtual machine...)

#TODO: debug occasional query document formatting errors in prefetch-and-prune
#TODO: figure out why accuracy is stable when splitting after preprocessing and unstable before

def distance(x1,x2):
    return np.sqrt( np.sum((np.array(x1) - np.array(x2))**2) )

#Get word centroid of each document (represent each document as a weighted average of its word vectors)
def getWordCentroid(document, nbow):
	#basically sum columns multiplied by corresponding entries in normalized bag of words
	return np.dot(document, nbow)

#Compute the Word Centroid Distance: distance between word centroids of documents
def wcd(doc1, doc1_nbow, doc2, doc2_nbow):
	return distance(getWordCentroid(doc1, doc1_nbow), getWordCentroid(doc2, doc2_nbow))

#Given a word and another document, find the word's nearest neighbor word vector in the other document
#NOTE: for demo/experimentation purposes only: used to calculate rWMD with 1 constraint
#to calculate rwmd as the max of relaxing either of the 2 constraints, use rWMD
# (in practice this is basically as efficient and performs better)
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
#NOTE: for demo/experimentation purposes only: used to calculate rWMD with 1 constraint
#to calculate rwmd as the max of relaxing either of the 2 constraints, use rWMD
# (in practice this is basically as efficient and performs better)
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
def rWMD(doc1, doc1_nbow, doc2, doc2_nbow):
	doc1_numWords = doc1.shape[1] #recall that words are in columns
	doc2_numWords = doc2.shape[1]
	doc1_nearestDists = [float("Inf")] * doc1_numWords #keep track of nearest neighbors of each word in document 1
	doc2_nearestDists = [float("Inf")] * doc2_numWords #keep track of nearest neighbors of each word in document 2
	for word1_index in range(doc1_numWords):
		word1 = doc1[:,word1_index]
		for word2_index in range(doc2_numWords):
			word2 = doc2[:,word2_index]
			wdist = distance(word1,word2) #distance between the word vectors
			#note: to compute rWMD we don't have to save the actual neighbors, just their distances
			if wdist < doc1_nearestDists[word1_index]: #found a new nearest neighbor of word 1
				doc1_nearestDists[word1_index] = wdist #save its distance as the new closest distance
			if wdist < doc2_nearestDists[word2_index]: #same thing for wrod 2
				doc2_nearestDists[word2_index] = wdist

	#compute document distances based on relaxing each constraint (i.e. with each nearest-neighbor search)
	#weight distance each word is from its nearest neighbor by the original word's occurrence weight
	rwmd1 = np.sum(doc1_nearestDists * doc1_nbow) #weighted sum of pairwise product
	rwmd2 = np.sum(doc2_nearestDists * doc2_nbow)
	rwmd = max(rwmd1, rwmd2) #take max of 2 relaxations of the formal WMD problem (see Kusner et. al, 2015)
	return rwmd 

#Compute exact WMD between two documents
def wmd(doc1, doc1_nbow, doc2, doc2_nbow):
	doc1 = doc1.T #for converting to list and feeding to EMD solver
	doc2 = doc2.T
	wmd_dist = emd((doc1.tolist(), doc1_nbow.tolist()), (doc2.tolist(), doc2_nbow.tolist()), distance)
	return wmd_dist

#Prefetch and prune algorithm to approximate kNN search
#Use heuristics such as wcd and rwmd to avoid computing (expensive) wmd distance to non-neighbors
#num_NNcandidates specifies how many candidates to potentially compute more expensive rwmd or wmd distance to
#The higher it is, the more provably accurate (but also slower) the method is
#In practice, for many applications error decreases most around num_NNcandidates = 2*num_neighbors (Kusner et. al, 2015)
def prefetch_and_prune(query_doc, query_nbow, train_data, train_nbow, num_neighbors, num_NNcandidates):
	numTrain = len(train_data)
	#num_NNcandidates must be at least the number of neighbors and at most the number of training documents
	if num_NNcandidates < num_neighbors:
		num_NNcandidates = num_neighbors
	elif num_NNcandidates > numTrain:
		num_NNcandidates = numTrain

	#sort training documents by their WCD distance from query document (cheap to compute)
	#implicitly sort them by sorting their indices
	sorted_train_indices = range(numTrain)
	sorted_train_indices.sort(key=lambda x: wcd(train_data[x], train_nbow[x], query_doc, query_nbow))

	#keep track of tentative k nearest neighbors
	nearest_neighbors = [None] * num_neighbors
	nearest_distances = [float("inf")] * num_neighbors

	#compute exact WMD distance to the first k of these (the tentative nearest neighbors)
	#for the remaining documents (up to the number specified by num_NNcandidates):
	#compute rwmd to training document
	#only need to compute exact wmd if this is smaller than exact wmd of k-th nearest neighbor
	for doc_index in range(num_NNcandidates):
		sorted_doc_index = sorted_train_indices[doc_index] #actual document index (not just its order when sorted by WCD)
		train_doc_words = train_data[sorted_doc_index] #the document (word vector representation) corresponding to that index
		train_doc_nbow = train_nbow[sorted_doc_index] #the document (normalized BOW representation) corresponding to that index
		rwmd_dist = 0 #clearly less than infinity, the default nearest neighbors, so we'll replace the first k NN with their wmds
		if doc_index >= num_neighbors: #when we want to try to prune with rwmd but still see if we need to calculate wmd
			rwmd_dist = rWMD(train_doc_words, train_doc_nbow, query_doc, query_nbow)
		if rwmd_dist < nearest_distances[0]: #either this is among the first k or it's a possible nearest neighbor
			#calculate wmd and see if it's among the k closest discovered so far
			neighbor_number = 0
			wmd_dist = wmd(train_doc_words, train_doc_nbow, query_doc, query_nbow)
			#insert each of the first 
			while wmd_dist < nearest_distances[neighbor_number]:
				neighbor_number += 1
				if neighbor_number >= len(nearest_distances):
					break
			if neighbor_number > 0: #we found a new nearest neighbor
				#insert it into the right place (so as to maintain sorted order among nearest neighbors)
				nearest_distances.insert(neighbor_number, wmd_dist)
				nearest_neighbors.insert(neighbor_number, sorted_doc_index)
				#remove old kth-NN
				nearest_neighbors.pop(0)
				nearest_distances.pop(0)
	return nearest_neighbors

#Given a query document and a training set, return k nearest neighbors
def kNN(query_doc, query_nbow, train_data, train_nbow, num_neighbors):
	#Keep track of nearest neighbors and their distances (initially we have none)
	nn_distances = [float("inf")] * num_neighbors #keep ordered list of k nearest neighbors, largest to smallest distance
	nearest_neighbors = [None] * num_neighbors #keep track of indices of nearest neighbors

	#Documents are entries in a list of training data (each document is a matrix whose columns are word vectors)
	for train_doc_index in xrange(0,train_data.size):
		train_doc = train_data[train_doc_index]
		train_doc_nbow = train_nbow[train_doc_index]
		dist = wcd(query_doc, query_nbow, train_doc, train_doc_nbow)
		#dist = rWMD(query_doc, query_nbow, train_doc, train_doc_nbow)
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
	num_train = train_data.size
	num_neighbors = min(num_neighbors, num_train) #can't request more neighbors than there are
	num_test = 100
	#num_test = test_data.size
	for query_index in xrange(0,num_test):
		query_doc = test_data[query_index]
		#documents can be empty if they contain only stopwords (we need to handle this, at least on the test end)
		if query_doc.shape[1] == 0: #empty document
			num_test -= 1 #don't count this document, and don't try to predict it (that's meaningless)
			continue
		query_nbow = test_nbow[query_index]
		#query_NN = kNN(query_doc,query_nbow,train_data,train_nbow,num_neighbors)
		query_NN = prefetch_and_prune(query_doc, query_nbow, train_data, train_nbow, num_neighbors, 4*num_neighbors)
		NN_labels = train_labels[query_NN]
		predicted_label = mode(NN_labels)[0][0] #get modal value
		if predicted_label == test_labels[query_index]:
			correct_preds += 1
		num_preds += 1
		if num_preds % 20 == 0: #print out every so often
			print("Made %d of %d predictions" % (num_preds, num_test)),
			if num_preds % 100 == 0: #print accuracy every 100th time
				print( "with accuracy %f" % (float(correct_preds) / num_preds))
			else:
				print
	accuracy = float(correct_preds) / num_test
	return accuracy

#If user wants to provide a big dump of data and split it into train and test here (instead of passing in a pre-defined split)
#Split randomly 80% train, 20% test
def splitTrainTest(data, nbow, labels):
	numData = data.shape[0]
	train_proportion = 0.8
	trainTest_order = range(numData)
	random.shuffle(trainTest_order) #actual indices of data/bow/labels are now in random order (take first 80% as train, rest as test)
	numTrain = int(train_proportion * numData)
	train_indices = trainTest_order[:numTrain]
	test_indices = trainTest_order[numTrain:]
	train_data = data[train_indices] 
	train_nbow = nbow[train_indices]
	train_labels = labels[train_indices]
	#remove empty documents from training (can be caught at preprocessing time "in the wild")
	nonempty = []
	for bow_index in range(train_nbow.size):
		if train_nbow[bow_index].size > 0: #document is nonempty
			nonempty.append(bow_index)
	nonempty_indices = np.asarray(nonempty)
	train_data = train_data[nonempty_indices]
	train_nbow = train_nbow[nonempty_indices]
	train_labels = train_labels[nonempty_indices]
	test_data = data[test_indices]
	test_nbow = nbow[test_indices]
	test_labels = labels[test_indices]
	return train_data, train_nbow, train_labels, test_data, test_nbow, test_labels
	
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

	if sys.argv[2] == "SPLIT": #user wants to split into training and test sets (and must have provided labels since they want to test)
		all_labels_file = sys.argv[4]
		with open(all_labels_file, "r") as alllf:
			all_labels = pickle.load(alllf)
		#recall we read in the "train" data already but it was actually all the data
		train_data, train_nbow, train_labels, test_data, test_nbow, test_labels = splitTrainTest(train_data, train_nbow, all_labels)
		test_accuracy = testWithLabels(num_neighbors, train_data, test_data, train_nbow, test_nbow, train_labels, test_labels)   
		print "Test accuracy: ", test_accuracy
		print
		
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
		print

	elif len(sys.argv) == 4: #user hasn't provided labels
		#so now we know test data is a single document
		with open(test_data_file, "r") as tedf:
			[test_data, test_nbow, CTe, wordsTe] = pickle.load(tedf)
		recommendations = kNN(test_data, test_nbow, train_data, train_nbow, num_neighbors)
		print "We recommend the following documents: ", recommendations

	elif not sys.argv[2] == "SPLIT": #rudimentary input error handling
		print "Usage: python approx_wmd.py [train_data_vectors] [test_data_vectors] [num_neighbors]"
		print "Optional: add [train_data_labels] [test_data_labels] for testing at the end"
