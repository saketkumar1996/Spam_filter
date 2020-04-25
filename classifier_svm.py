from __future__ import print_function, division
import nltk
import os
import random
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify
from sklearn.svm import SVC
import numpy as np

stoplist = stopwords.words('english')

def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        f = open(folder + a_file, 'r')
        a_list.append(f.read())
    f.close()
    return a_list

def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(unicode(sentence, errors='ignore'))]

def get_dictionary(all_emails):
	emails = [a[0] for a in all_emails]
	all_words = []
	for mail in emails:
		all_words += mail.split()
	dictionary = Counter(all_words)

	list_to_remove = dictionary.keys()
	for item in list_to_remove:
		if item.isalpha() == False:
			del dictionary[item]
		elif len(item) == 1:
			del dictionary[item]

	dictionary = dictionary.most_common(2500)
	return dictionary

def extract_feature(dictionary, words):
	temp = words.split()
	dictionary_words = [a[0] for a in dictionary]
	features = [words.count(word) if word in temp else 0 for word in dictionary_words]
	return features

def train(features, samples_proportion):
	print("training")
	train_size = int(len(features) * samples_proportion)
	train_set, test_set = features[:train_size], features[train_size:]
	print ('Training set size = ' + str(len(train_set)) + ' emails')
	print ('Test set size = ' + str(len(test_set)) + ' emails')
	x_train, y_train = [a[0] for a in train_set], [b[1] for b in train_set]
	classifier = SVC(kernel="rbf",C=120, verbose=True)
	classifier.fit(x_train, y_train)
	return train_set, test_set, classifier

if __name__ == "__main__":
	# initialise the data
    spam = init_lists('data/spam/')
    ham = init_lists('data/ham/')
    all_emails = [(email, 1) for email in spam]
    all_emails += [(email, 0) for email in ham]
    random.shuffle(all_emails)
    print("Total data points ",len(all_emails))

    dictionary = get_dictionary(all_emails)

    print("Extracting features")

    all_features = [(extract_feature(dictionary, email), label) for (email, label) in all_emails]

    print("Features extracted")

    train_set, test_set, classifier = train(all_features, 0.8)

    x_test, y_test = [a[0] for a in test_set], [b[1] for b in test_set]

    print("score of trained classifier ", classifier.score(x_test, y_test))