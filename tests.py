import numpy as np
import pandas as pd
import spacy
import statistics

# Load Data CSV
data = pd.read_csv('data/deceptive-opinion.csv')

#Finding statistical measures for the data length 
tok = spacy.blank('en')
lengths = []

for review in data['text']:
    lengths.append(len(tok(review)))
total = sum(lengths)

mean = statistics.mean(lengths)
median = statistics.median(lengths)
mode = statistics.mode(lengths)
stdev = statistics.stdev(lengths)
min_ = min(lengths)
max_ = max(lengths)

print(f'Mean = {mean}')
print(f'median = {median}')
print(f'mode = {mode}')


#find the percentiles of the data lenghts
print(np.percentile(lengths, [90, 95, 99])) # ouputs [295.1  364.1  509.04]
#Thus, I'll use 364 as my max length so 95% of data won't be cut off
data['length'] = lengths
print(data.groupby('deceptive')['length'].describe())


#Split data by group
truthful_positive = data[0:400]
truthful_negative = data[800:1200]

deceptive_positive = data[400:800]
deceptive_negative = data[1200:1600]
