import gensim, pdb, sys, scipy.io as io, numpy as np, pickle, string, random

# read datasets line by line
def read_line_by_line(dataset_name,C,model,vec_size):
	# get stop words (except for twitter!)
	SW = set()
	for line in open('stop_words.txt'):
		line = line.strip()
		if line != '':
			SW.add(line)

	stop = list(SW)


	f = open(dataset_name)
	if len(C) == 0:
		C = np.array([], dtype=np.object)
	num_lines = sum(1 for line in open(dataset_name))
	y = np.zeros((num_lines,))
	X = np.zeros((num_lines,), dtype=np.object)
	BOW_X = np.zeros((num_lines,), dtype=np.object)
	count = 0
	remain = np.zeros((num_lines,), dtype=np.object)
	the_words = np.zeros((num_lines,), dtype=np.object)
	for line in f:
		#print '%d out of %d' % (count+1, num_lines)
		line = line.strip()
		line = line.translate(string.maketrans("",""), string.punctuation)
		T = line.split('\t')
		classID = T[0]
		if classID in C:
			IXC = np.where(C==classID)
			y[count] = IXC[0]+1
		else:
			C = np.append(C,classID)
			y[count] = len(C)
		W = line.split()
		F = np.zeros((vec_size,len(W)-1))
		#F = np.zeros((len(W)-1, vec_size))
		
		inner = 0
		RC = np.zeros((len(W)-1,), dtype=np.object)
		#word_order = np.zeros((len(W)-1), dtype=np.object)
		word_order = np.empty((len(W) - 1), dtype=np.object)
		word_order.fill('')
		bow_x = np.zeros((len(W)-1,))
		for word in W[1:len(W)]:
			try:
				test = model[word]
				if word in stop:
					#word_order[inner] = ''
					continue
				if word in word_order:
					IXW = np.where(word_order==word)
					bow_x[IXW] += 1
					#word_order[inner] = ''
				else:
					word_order[inner] = word
					bow_x[inner] += 1
					F[:,inner] = model[word]
					#F[inner,:] = model[word]
			except KeyError, e:
				#print 'Key error: "%s"' % str(e)
				word_order[inner] = ''
			inner = inner + 1
		Fs = F.T[~np.all(F.T == 0, axis=1)]
		#Fs = F[~np.all(F == 0, axis=1)]
		word_orders = word_order[word_order != '']
		bow_xs = bow_x[bow_x != 0]
		nbow_xs = normalize_bow(bow_xs) #Added by MH
		X[count] = Fs.T
		the_words[count] = word_orders
		BOW_X[count] = nbow_xs #bow_xs
		count = count+ 1
	return (X,BOW_X,y,C,the_words)

def save_data(dataset,save_file_data,save_file_labels,model,vec_size):
	# 2. read document data
	(X,BOW_X,y,C,words)  = read_line_by_line(dataset,[],model,vec_size)

	# 3. save pickle of extracted variables
	with open(save_file_data, 'w') as f:
		pickle.dump([X, BOW_X, C, words], f)

	with open(save_file_labels, 'w') as labels_file:
		pickle.dump(y,labels_file)

#Take a bag of words and normalize its entries so that they sum to 1
def normalize_bow(bow):
	return bow / float(np.sum(bow))
	
def main():
	# 0. load word2vec model (trained on Google News)
	#model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
	model = gensim.models.Word2Vec.load("GoogleNews_vectors")
	vec_size = 300

	#handle input argument errors
	if len(sys.argv) != 3: #wrong number of arguments
		print "Usage: python process_data.py [train_data_text_file] [test_data_text_file]"

	# 1. specify train/test datasets
	train_dataset = sys.argv[1] # e.g.: 'twitter.txt'
	test_dataset = sys.argv[2] #e.g. twitter_test.txt'
	train_name = train_dataset.split(".")[0]
	test_name = test_dataset.split(".")[0]
	
	train_data_file = train_name + ".pk"
	test_data_file = test_name + ".pk"
	train_labels_file = train_name + "_labels.pk"
	test_labels_file = test_name + "_labels.pk"

	# 2. read and save document data
	save_data(train_dataset,train_data_file,train_labels_file,model,vec_size)
	save_data(test_dataset,test_data_file,test_labels_file,model,vec_size)
	
if __name__ == "__main__":
	main()						                                                                     
