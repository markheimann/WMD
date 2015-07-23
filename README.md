# WMD

Implementation of the Word Mover's Distance metric for document distances.
Based on (Kusner et. al, 2015): http://www.cse.wustl.edu/~kilian/papers/wmd_metric.pdf
Some preprocessing code + WMD solver from code at matthewkusner.com. 
Rewrote much of the rest of the code, implemented other heuristics described in the paper,
  and reworked it for my own purposes
Used a quick kNN implementation based on bag of words and sklearn KNN classifier and some
  basic statistical tests to suggest that WMD does in fact do appreciably better
  
Notes on usage: 
You need NumPy and SciPy as well as Gensim, a Python library with many NLP tools (https://radimrehurek.com/gensim/).  You also need a word2vec model to learn word representations--this code uses representations learned from a Google News corpus of ~100 billion words, made freely available at https://code.google.com/p/word2vec/.  In particular, this code uses a saved Gensim model that had been trained on these vectors.  To do the same, import gensim and use the command load_word2vec_format() with the word vectors and save that model so that it can be loaded somewhat more quickly back in.  

You can preprocess your data (removing stop words and words with no word2vec embedding) as well as represent the words in your data with their word2vec representations by running: python process_data.py [data_file].  The ability to process the training and test data separately is in the pipeline.  This will save pickle files with the word2vec representations of the data and the labels.  The files will have the same name as the original data file, with "_data" or "_labels" appended and a .pk extension.  

WMD with all of the approximations mentioned in the paper is implemented in approx_wmd.py.  You can perform kNN search with the WMD distance metric with the following command: python process_data.py [data_file] SPLIT num_neighbors [labels_file].  SPLIT means that the data file will be split into training and test data randomly with an 80/20 train/test ratio.  num_neighbors is the number of neighbors for the k-nearest neighbors search, and the labels file is the file containing the labels corresponding to the data (which will also be split to match the split into training and test of the data).   
