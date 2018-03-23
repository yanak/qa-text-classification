from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import MeCab
import json
import numpy as np
from keras.models import load_model
import tensorflow as tf
from flask import Flask, render_template, request
import csv
import re

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        return render_template('app.html')
    else:
        return render_template('app.html', result=str(predict(request.form['question'])))

def tokenize(text):
	wakati = MeCab.Tagger("-O wakati")
	wakati.parse("")
	words = wakati.parse(text)
   
	# Make word list
	if words[-1] == u"\n":
        	words = words[:-1]
        
	return words

def filter(text):
    # remove unnecessary characters
    result = re.compile('-+').sub('', text)
    result = re.compile('[0-9]+').sub('0', result)
    result = re.compile('\s+').sub('', result)
    
    return result

def predict(text):
    
    # load questions
	questions = []
	with open("../data/questions.tsv", 'r', encoding="utf-8") as tsvin:
		tsvin = csv.reader(tsvin, delimiter='\t')
		for row in tsvin:
        		columns = []
        		columns.append(row[1])
        		columns.append(row[2])
        		columns.append(row[3])
        		
        		questions.append(columns)

    # drop the first column
	sub_texts = []
	for t in questions:
		columns = []
		columns.append(filter(t[0]))
		columns.append(t[1])
		columns.append(t[2])
		
		if len(filter(t[0])) > 0:
			sub_texts.append(columns)

    # drop colum names
	sub_texts.pop(0)

    # Create samples and labels
	labels = []
	texts = []
	threashold = 700
	cnt1 = 0
	cnt2 = 0
	cnt3 = 0
	for i, row in enumerate(sub_texts):
	    if 'Account' in row[2]:
	        if cnt2 < threashold:
	            cnt2 += 1
	            labels.append(2)
	            texts.append(row[0])
	    elif 'Payment' in row[2]:
	        if cnt3 < threashold:
	            cnt3 += 1
	            labels.append(3)
	            texts.append(row[0])
	    else:
	        if cnt1 < threashold:
	            cnt1 += 1
	            texts.append(row[0])
	            labels.append(1)

	print("labels size:%d" % len(labels))

    # create word index
	texts = [tokenize(a) for a in texts]
	maxlen = 1000
	max_words = 15000
	tokenizer = Tokenizer(num_words=max_words)
	tokenizer.fit_on_texts(texts)
	tokenized_text = tokenize(filter(text))
	seq = tokenizer.texts_to_sequences([tokenized_text])
	padded_seq = pad_sequences(seq, maxlen=maxlen)

    # predict
    model = load_model('../pre_trained_cs_model.h5')
    res = model.predict([padded_seq])
    
    return np.argmax(res[0])
