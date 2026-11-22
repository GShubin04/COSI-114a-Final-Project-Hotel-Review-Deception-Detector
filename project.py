import numpy as np
import pandas as pd
import spacy

data = pd.read_csv('data/preprocessed-reviews.csv')
tok = spacy.load("en_core_web_sm", disable = ["ner"])

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



def extract_features(texts, batch_size = 64):
    rows = []
    for review in tok.pipe(texts, batch_size = batch_size):
        #continue here once I figure out if I can use spacy
        print("to be continued...")