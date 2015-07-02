# WMD

Implementation of the Word Mover's Distance metric for document distances.
Based on (Kusner et. al, 2015): http://www.cse.wustl.edu/~kilian/papers/wmd_metric.pdf
Some preprocessing code + WMD solver from code at matthewkusner.com. 
Rewrote much of the rest of the code, implemented other heuristics described in the paper,
  and reworked it for my own purposes
Used a quick kNN implementation based on bag of words and sklearn KNN classifier and some
  basic statistical tests to suggest that WMD does in fact do appreciably better
  
  
