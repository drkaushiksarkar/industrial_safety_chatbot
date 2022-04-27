from flask import Flask,render_template,url_for,request
import os
import pandas as pd
import numpy as np
import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import SpatialDropout1D
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
import numpy as np

import tensorflow as tf
import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import SpatialDropout1D
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

import pickle


import random
import copy
import time
import pandas as pd
import numpy as np
import gc
import re
import torch

#import spacy
from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')
from collections import Counter

from nltk import word_tokenize

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from sklearn.metrics import f1_score
import os 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# cross validation and metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.optim.optimizer import Optimizer

from sklearn.preprocessing import StandardScaler
from multiprocessing import  Pool
from functools import partial
import numpy as np
from sklearn.decomposition import PCA
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import scikitplot as skplt

import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST', 'GET'])

def predict():
	
	def clean_text(text):
		REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
		BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
		STOPWORDS = set(stopwords.words('english'))
		text = text.lower() # lowercase text
		text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
		text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
		text = text.replace('x', '')
		text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
		return text
	contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
	def _get_contractions(contraction_dict):
		contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
		return contraction_dict, contraction_re
	contractions, contractions_re = _get_contractions(contraction_dict)
	def replace_contractions(text):
		def replace(match):
			return contractions[match.group(0)]
		return contractions_re.sub(replace, text)
	
	# The maximum number of words to be used. (most frequent)
	MAX_NB_WORDS = 5000
	# Max number of words in each complaint.
	MAX_SEQUENCE_LENGTH = 50
	# This is fixed.
	EMBEDDING_DIM = 100
	
	def load_glove(word_index):
		EMBEDDING_FILE = 'glove.6B.100d.txt'
		def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
		embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
		all_embs = np.stack(embeddings_index.values())
		emb_mean,emb_std = -0.005838499,0.48782197
		embed_size = all_embs.shape[1]
		nb_words = min(MAX_NB_WORDS, len(word_index)+1)
		embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
		for word, i in word_index.items():
			if i >= MAX_NB_WORDS: continue
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None: 
				embedding_matrix[i] = embedding_vector
			else:
				embedding_vector = embeddings_index.get(word.capitalize())
				if embedding_vector is not None: 
					embedding_matrix[i] = embedding_vector
		return embedding_matrix
		
	
	
	class BiLSTM(nn.Module):
		def __init__(self):
			super(BiLSTM, self).__init__()
			self.hidden_size = 64
			drp = 0.2
			n_classes = len(le.classes_)
			self.embedding = nn.Embedding(MAX_NB_WORDS, EMBEDDING_DIM)
			self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
			self.embedding.weight.requires_grad = True
			self.lstm = nn.LSTM(EMBEDDING_DIM, self.hidden_size, bidirectional=True, batch_first=True)
			self.linear = nn.Linear(self.hidden_size*4 , 64)
			self.relu = nn.ReLU()
			self.dropout = nn.Dropout(drp)
			self.out = nn.Linear(64, n_classes)
		def forward(self, x):
			#print(x.size())
			h_embedding = self.embedding(x)
			#_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
			h_lstm, _ = self.lstm(h_embedding)
			avg_pool = torch.mean(h_lstm, 1)
			max_pool, _ = torch.max(h_lstm, 1)
			conc = torch.cat(( avg_pool, max_pool), 1)
			conc = self.relu(self.linear(conc))
			conc = self.dropout(conc)
			out = self.out(conc)
			return out
	
	if request.method == 'POST':
		
		new_complaint = [request.form['message']]
		tokenizer = pickle.load(open("tokenizer.pickle", "rb"))
		seq = tokenizer.texts_to_sequences(new_complaint)
		padded = pad_sequences(seq, maxlen=50)
		device = torch.device('cpu')
		le = pickle.load(open("le", "rb"))
		embedding_matrix = pickle.load(open("embedding_matrix", "rb"))
		model = BiLSTM()
		model.load_state_dict(torch.load('bilstm.h5', map_location=device))
		padded = torch.from_numpy(padded)
		pred = model(padded).detach()
		val_pred = F.softmax(pred).cpu().numpy()
		my_prediction =[le.classes_[x] for x in val_pred.argmax(axis=1)]
		
		Date = request.form['Date']
		Country = request.form['Country']
		Local = request.form['Local']
		Gender = request.form['Gender']
		Employee = request.form['Employee Type']
		Risk = request.form['Critical Risk']
		Industry = request.form['Industry Sector']
		new_data = {'Country': Country,
        'Local':Local,
        'Industry':Industry,
        'Gender':Gender,
        'Employee':Employee,
        'Risk':Risk,
        'Date':Date}
		new_df = pd.DataFrame(new_data, index=[0])
		new_df['Date'] = pd.to_datetime(new_df['Date'])
		new_df['Month'] = new_df['Date'].apply(lambda x : x.month_name())
		new_df['Weekday'] = new_df['Date'].apply(lambda x : x.day_name())
		
		new_df['Employee'] = new_df['Employee'].str.replace(' ', '_')
		new_df['Risk'] = new_df['Risk'].str.replace('\n', '').str.replace(' ', '_')
		new_df.drop('Date', inplace=True, axis=1)
		encoder = pickle.load(open("encoder", "rb"))
		test_preprocessed = encoder.transform(new_df)
		feature_list = pickle.load(open("feature_list", "rb"))
		test_preprocessed = pd.DataFrame(test_preprocessed)
		test_preprocessed = test_preprocessed[feature_list]
		finalized_ml_model = pickle.load(open("finalized_ml_model.sav", "rb"))
		
		my_prediction_2 = finalized_ml_model.predict(test_preprocessed)
		
	return render_template('result.html', prediction = my_prediction, Day = new_df['Weekday'][0], prediction_2 = my_prediction_2)



if __name__ == '__main__':
	app.run(debug=True)