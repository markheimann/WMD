from sklearn.neighbors import KNeighborsClassifier
import numpy as np, sys, random

#Read in data in file name for preprocessing and use: assumes data format is label separated by tab from words
#Also lowercase all words in case they aren't already (which in the twitter data, for example, they already are)
def process_data(file_name):
	stops_file = open("stop_words.txt", "r") #each (unique) stop word is on its own line, with extra white space for padding
	stop_words = set([line.strip() for line in stops_file if line != "\n"]) #strip off carriage returns and store stop words in a list
	
	datafile = open(file_name, "r") #open the file containing the data for reading
	text_labels = [] #store labels as they are read in
	text_data = [] #store data as it is read in

	for datum in datafile: #each line represents a piece of data
		split = datum.split("\t")
		label = split[0].replace("\"", "") #label is separated by a tab (also get rid of quotation marks)
		text = split[1].lower().split() #words in document are on the other side. lowercase and store in list
		preproc_text = [word for word in datum if word not in stop_words] #remove stop words and leave what's left
		text_labels.append(label)
		text_data.append(text)

	#vectorize data and labels so that they can be fed into a machine learning algorithm
	return vectorize(text_data), vectorize(text_labels)

#Convert a list of (e.g.) textual features/labels into numbers so that it can be fed into a machine learning algorithm
def vectorize(data):
	if type(data[0]) is str: #data is a list (e.g. of labels) and not a list of lists (e.g. of features)
		return vectorize_list(data)

	vocabulary = dict()
	vocab_size = 0
	vectorized_docs = list() #list of all the vectorized documents
	for doc_wordList in data:
		document = [0] * vocab_size
		for word in doc_wordList:
			if word not in vocabulary: #we've seen a new word
				vocab_size += 1
				vocabulary[word] = vocab_size
				document.append(0) #keep expanding document size to match size of vocabulary seen so far 
			feature_number = vocabulary[word] #we know this is in the dictionary because if not we just added it
			document[feature_number - 1] += 1
		vectorized_docs.append(document)
	
	#Make sure all documents have the same size as the vocabulary (otherwise add 0's on to the end)
	for doc in vectorized_docs:
		while len(doc) < vocab_size:
			doc.append(0)
	
	#Combine vectorized documents into a Numpy array
	vec_data = np.asarray(vectorized_docs)
	return vec_data

def vectorize_list(data):
	unique_entries = dict()
	num_unique = 0
	vectorized_features = list()
	for entry in data:
		if entry not in unique_entries: #we've seen a new word
			num_unique += 1
			unique_entries[entry] = num_unique
		feature_number = unique_entries[entry] #we know this is in the dictionary because if not we just added it
		vectorized_features.append(feature_number)
	return np.asarray(vectorized_features)

def splitTrainTest(data, labels):
	train_fraction = 0.8
	numData = data.shape[0]
	numTrain = int(numData * train_fraction)
	ordering = range(numData)
	random.shuffle(ordering)
	train_indices = ordering[:numTrain]
	test_indices = ordering[numTrain:]
	train_data = data[train_indices,:]
	train_labels = labels[train_indices]
	test_data = data[test_indices,:]
	test_labels = labels[test_indices]
	return train_data, train_labels, test_data, test_labels

#actually take in training/test data and direct it appropriately to be preprocessed and used for classification
if __name__ == "__main__":
	data_file = sys.argv[1]
	all_data, all_labels = process_data(data_file)
	train_data, train_labels, test_data, test_labels = splitTrainTest(all_data, all_labels)

	clf = KNeighborsClassifier(n_neighbors=5)
	clf.fit(train_data, train_labels)
	test_preds = clf.predict(test_data)
	accuracy = np.mean(test_preds == test_labels)
	print "Accuracy: ", accuracy
