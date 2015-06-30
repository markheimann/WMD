import sys, random

filePath = sys.argv[1]
randomize = True #by default we randomly shuffle data
if len(sys.argv) == 3: #user specified instructions vis-a-vis random shuffling of data
	randomize = sys.argv[2] #set randomize to user-specified parameter

fileName = filePath.split(".")[0]
train_proportion = 0.8

data_file = open(filePath, "r")
data = data_file.readlines()
numData = len(data)

#take first part (specified by train proportion) as training data, rest as test
numTrain = int(train_proportion * numData)

#if randomized shuffle the data randomly
if randomize:
	random.shuffle(data)

train_data = data[:numTrain]
test_data = data[numTrain:]

#write to respective files
train_fileName = fileName + "_train.txt"
test_fileName = fileName + "_test.txt"

train_file = open(train_fileName, "w")
for datum in train_data:
	train_file.write(datum)
train_file.close()

test_file = open(test_fileName, "w")
for datum in test_data:
	test_file.write(datum)
test_file.close()
