import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def lemmatize(words):
    return [lemmatizer.lemmatize(w.lower()) for w in words]

def bag_of_words(sentence, patterns):
    sentence_words = lemmatize(tokenize(sentence))
    bag = [0] * len(patterns)
    for idx, word in enumerate(patterns):
        if word in sentence_words:
            bag[idx] = 1
    return np.array(bag)