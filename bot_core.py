import json
import random
import numpy as np
from nltk_utils import bag_of_words, tokenize, lemmatize

# Load intent data
with open("intents.json") as file:
    data = json.load(file)

# Build vocabulary
all_words = []
tags = []
xy = []

for intent in data["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Clean word list
all_words = lemmatize(all_words)
all_words = sorted(set([w for w in all_words if w.isalpha()]))

# Predict tag
def predict_tag(user_input):
    bow = bag_of_words(user_input, all_words)
    scores = {}
    for i, (pattern, tag) in enumerate(xy):
        score = np.dot(bow, bag_of_words(" ".join(pattern), all_words))
        if tag in scores:
            scores[tag] += score
        else:
            scores[tag] = score
    return max(scores, key=scores.get) if scores else "default"

# Get response
def get_response(user_input):
    tag = predict_tag(user_input)
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't get that."