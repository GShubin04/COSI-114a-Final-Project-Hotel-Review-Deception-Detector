import numpy as np
import pandas as pd
import spacy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = pd.read_csv('data/preprocessed-reviews.csv')
tok = spacy.load("en_core_web_sm", disable = ["parser", "ner"])
tok.add_pipe('sentencizer')
#Split data by group
truthful_positive = data[0:400]
truthful_negative = data[800:1200]
deceptive_positive = data[400:800]
deceptive_negative = data[1200:1600]

#words for building features
first_sing = {"i", "me", "my", "mine", "myself"}
first_plur = {"we", "us", "our", "ours", "ourselves"}
third = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"}
spatial_words = {"in", "on", "at", "near", "by", "next", "across", "between", "behind", "above", 
    "below", "inside", "outside", "floor", "lobby", "street", "room", "stairs"}
temporal_words = {"when", "while", "during", "after", "before", "since", "until", "norning",
    "afternoon", "evening", "night", "mightnight", "weekend", "monday",
    "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday" }
negations = {"no", "not", "never", "neither", "nobody", "nothing", "nowhere", "hardly", "barely"}
modals = {"would", "could", "should", "might", "may", "must", "maybe", "perhaps", "seem", "seemed", "apparently"}
intensifiers = {"very", "really", "extremely", "absolutely", "completely", "totally", "definitely", "truly", 
    "amazing", "horrible", "terrible", "worst", "best"}



def extract_features(review, data = data, batch_size = 64):
    rows = []
    i = 0
    for doc in tok.pipe(review, batch_size = 64):
        polarity = data["polarity"][i]
        tokens = [t for t in doc if not t.is_space]
        words = [t for t in doc if t.is_alpha]


        n_tokens = max(len(tokens), 1)
        n_words = max(len(words), 1)
        lower_words = [word.text.lower() for word in words]
        lower_tokens = [t.text.lower() for t in tokens]
        sents = list(doc.sents)
        n_sents = max(len(sents), 1)

        features = {
            #POS distributions from spaCy (all tags from universal dependencies POS list)
            "adj_rate" : sum(t.pos_  == "ADJ" for t in tokens) / n_tokens,
            "adposition_rate" : sum(t.pos_  == "ADP" for t in tokens) / n_tokens,
            "adv_rate" : sum(t.pos_  == "ADV" for t in tokens) / n_tokens,
            "aux_rate" : sum(t.pos_  == "AUX" for t in tokens) / n_tokens,
            "coord_conj_rate": sum(t.pos_  == "CCONJ" for t in tokens) / n_tokens,
            "det_rate" : sum(t.pos_  == "DET" for t in tokens) / n_tokens,
            "interjection_rate" : sum(t.pos_  == "INTJ" for t in tokens) / n_tokens,
            "noun_rate" : sum(t.pos_  == "NOUN" for t in tokens) / n_tokens,
            "num_rate" : sum(t.pos_  == "NUM" for t in tokens) / n_tokens,
            "part_rate" : sum(t.pos_  == "PART" for t in tokens) / n_tokens,
            "pronoun_rate" : sum(t.pos_  == "PRON" for t in tokens) / n_tokens,
            "prop_noun_rate" : sum(t.pos_  == "PROPN" for t in tokens) / n_tokens,
            "punctuation_rate" : sum(t.pos_  == "PUNCT" for t in tokens) / n_tokens,
            "sub_conj_rate" : sum(t.pos_  == "SCONJ" for t in tokens) / n_tokens,
            "symbol_rate" : sum(t.pos_  == "SYM" for t in tokens) / n_tokens,
            "verb_rate" : sum(t.pos_  == "VERB" for t in tokens) / n_tokens,
            "other_rate" : sum(t.pos_  == "X" for t in tokens) / n_tokens,

            #Specific Pronoun USage (deitic or otherwise)
            "1sg_rate" : sum(w in first_sing for w in lower_words) / n_words,
            "1pl_rate" : sum(w in first_plur for w in lower_words) / n_words,
            "3prn_rate" : sum(w in third for w in lower_words) / n_words,

            #the other sets I created
            "spatial_rate" : sum(w in spatial_words for w in lower_words) / n_words,
            "temporal_rate" : sum(w in temporal_words for w in lower_words) / n_words,
            "neg_rate" : sum(t in negations or t == "n't" for t in lower_tokens) / n_tokens,
            "modal_rate" : sum(w in modals for w in lower_words) / n_words,
            "intens_rate" : sum(w in intensifiers for w in lower_words) / n_words,

            #Other psycholinguistic cues
            "all_caps_rate" : sum(t.is_upper and len(t.text) > 1 for t in tokens) / n_tokens,
            "exclamation_rate" : sum(t.text == "!" for t in tokens) / n_sents,
            "question_rate" : sum(t.text == "?" for t in tokens) / n_sents,
            "avg_sent_len" : n_tokens / n_sents,
            "polarity" : 1 if polarity == "positive" else -1,
            #If features seem like they aren't fitting, maybe I'll add the rate at which they mention the Hotel's name
        }
        i += 1
        rows.append(features)
    return rows

#Splits
feature_dicts = extract_features(data['text'])
y = (data["deceptive"] == "deceptive").astype(int) #gold labels

strat_key = data["deceptive"] + "_" + data["polarity"]

#split into train, dev, and test (random state is my lucky number 22)
train_dev_split, test_dicts, y_train_dev, y_test, strat_train_dev, _ = train_test_split(
    feature_dicts, y, strat_key, test_size=.15, random_state= 22, stratify= strat_key) 
train_dicts, dev_dicts, y_train, y_dev = train_test_split(
    train_dev_split, y_train_dev, test_size= .15/.85, random_state=22, stratify= strat_train_dev)  

#Vectorize and fit the train, test, dev
vect = DictVectorizer()
train = vect.fit_transform(train_dicts)
dev = vect.transform(dev_dicts)
test = vect.transform(test_dicts)

rf = RandomForestClassifier(n_estimators= 300, random_state = 22)
rf.fit(train, y_train)
dev_yhat = rf.predict(dev)

conf_matrix = confusion_matrix(y_dev, dev_yhat)
print(conf_matrix)

rf_a = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Deceptive", "Truthful"])
rf_a.plot(cmap=plt.cm.BuPu)
plt.show()

