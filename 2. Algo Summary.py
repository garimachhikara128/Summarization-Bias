import numpy as np
import pandas as pd
import nltk 
from sklearn.feature_extraction.text import CountVectorizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from collections import Counter
from functools import reduce

import warnings
warnings.filterwarnings("ignore")

DATASET = 'US_Election'
LIMIT = 100 # US Election Dataset

def lin(V, m, lamb):
        '''DSDR with linear reconstruction
        Parameters
        ==========
        - V : 2d array_like, the candidate data set
        - m : int, the number of sentences to be selected
        - lamb : float, the trade off parameter
        Returns
        =======
        - L : list, the set of m summary sentences indices
        '''
        L = []
        V = np.array(V)
        B = np.dot(V, V.T) / lamb
        n = len(V)
        for t in range(m):
            scores = []
            for i in range(n):
                score = np.sum(B[:,i] ** 2) / (1. + B[i,i])
                scores += [(score, i)]
            max_score, max_i = max(scores)
            L += [max_i]
            B = B - np.outer(B[:,max_i], B[:,max_i]) / (1. + B[max_i,max_i])
        return L

    
def non(V, gamma, eps=1.e-8):
        '''DSDR with nonnegative linear reconstruction     
        Parameters
        ==========
        - V : 2d array_like, the candidate sentence set
        - gamma : float, > 0, the trade off parameter
        - eps : float, for converge
        Returns
        =======
        - beta : 1d array, the auxiliary variable to control candidate sentences
            selection
        '''
        V = np.array(V)
        n = len(V)
        A = np.ones((n,n))
        beta = np.zeros(n)
        VVT = np.dot(V, V.T) # V * V.T
        np.seterr(all='ignore')
        while True:
            _beta = np.copy(beta)
            beta = (np.sum(A ** 2, axis=0) / gamma) ** .5
            while True:
                _A = np.copy(A)
                A *= VVT / np.dot(A, VVT + np.diag(beta))
                A = np.nan_to_num(A) # nan (zero divide by zero) to zero
                if np.sum(A - _A) < eps: break
            if np.sum(beta - _beta) < eps: break
        return beta

# 1. Load the data
data = pd.read_csv('Data/' + DATASET + '/New_Input.txt', sep = '<\|\|>', header=None, engine = 'python')
data.columns = ['user_id', 'user_name', 'tweet_id', 'type', 'text']

# print(data.shape)
# print(data.head())

# 2. Clean the data
data['clean_text'] = data['text'].str.replace(r"@\S+", "") 
data['clean_text'] = data['clean_text'].str.replace(r"http\S+", "") 
data['clean_text'] = data['clean_text'].str.replace("[^a-zA-Z]", " ") 

# data['clean_text'] = data['text']

# nltk.download('stopwords')
stopwords=nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    clean_text=' '.join([word for word in text.split() if word not in stopwords])
    return clean_text

data['clean_text'] = data['clean_text'].apply(lambda text : remove_stopwords(text.lower()))

data['clean_text'] = data['clean_text'].apply(lambda x: x.split())

from nltk.stem.porter import * 
stemmer = PorterStemmer() 
data['clean_text'] = data['clean_text'].apply(lambda x: [stemmer.stem(i) for i in x])

data['clean_text'] = data['clean_text'].apply(lambda x: ' '.join([w for w in x]))

# print(data.head())

# 3. Convert word to vectors
count_vectorizer = CountVectorizer(stop_words='english') 
cv = count_vectorizer.fit_transform(data['clean_text'])
# print(cv.shape)
V = cv.toarray()
# type(V)

# 4. DSDR Linear
L = lin(V, m=LIMIT, lamb=0.1)
# print(L)

columns=count_vectorizer.get_feature_names_out()

file_dsdr_lin = open('Data/' + DATASET + '/DSDRLin.txt', 'w+')

for i in L:
  file_dsdr_lin.write(data.iloc[i]['text'] + "\n") 

file_dsdr_lin.close()

# 5. DSDR Non Linear
beta = non(V, gamma=0.1)

file_dsdr_non_lin = open('Data/' + DATASET + '/DSDRNonLin.txt', 'w+')

i = 1
for score, v, content in sorted(zip(beta, V, data['text']), reverse=True, key = lambda x: x[0]):

  if i > LIMIT :
    break

  file_dsdr_non_lin.write(content + "\n") 
  i += 1 

file_dsdr_non_lin.close()

# 6. Lex Rank

summarizer_lex = LexRankSummarizer()

s = ''  
for ps in data['clean_text'] :
  s += ps + '\n\n'

parser = PlaintextParser.from_string(s, Tokenizer("english"))

# Summarize using sumy LexRank
summary_lex = summarizer_lex(parser.document, LIMIT)

file_lex_rank = open('Data/' + DATASET + '/LexRank.txt', 'w+')

for sentence in summary_lex :
    row_no = data[data['clean_text'] == str(sentence)].index[0]
    file_lex_rank.write(data.iloc[row_no]['text'] + "\n") 

file_lex_rank.close()

# 7. LSA

summarizer_lsa = LsaSummarizer()

# Summarize using sumy LSA
summary_lsa = summarizer_lsa(parser.document, LIMIT)

file_lsa = open('Data/' + DATASET + '/LSA.txt', 'w+')

for sentence in summary_lsa :

    row_no = data[data['clean_text'] == str(sentence)].index[0]
    file_lsa.write(data.iloc[row_no]['text'] + "\n") 

file_lsa.close()

# 8. LUHN
summarizer_luhn = LuhnSummarizer()
summary_luhn = summarizer_luhn(parser.document,LIMIT)

file_luhn = open('Data/' + DATASET + '/LUHN.txt', 'w+')

for sentence in summary_luhn :

    row_no = data[data['clean_text'] == str(sentence)].index[0]
    file_luhn.write(data.iloc[row_no]['text'] + "\n") 

file_luhn.close()

# 9. SUM BASIC

def count_words(texts):
	"""
	Counts the words in the given texts, ignoring puncuation and the like.
	@param texts - Texts (as a single string or list of strings)
	@return Word count of texts
	"""

	if type(texts) is list:
		return len(texts)

	return len(texts.split())

def sumbasic_summarize(limit, data, update):  
  
    data_list = data.apply(lambda x: x.split())

    # Counter for all words.
    cnts = Counter()
    for sent in data_list :
        cnts += Counter(sent)

    # Number of tokens.
    N = float(sum(cnts.values()))

    # Unigram probabilities.
    probs = {w: cnt / N for w, cnt in cnts.items()}

    # print(probs)

    # List of all sentences in all documents.
    sentences = data.tolist()

    # print(len(sentences))

    summary = []
    # Add sentences to the summary until there are no more sentences or word
    # limit is exceeded.
    while len(sentences) > 0 and len(summary) < limit:
        # Track the max probability of a sentence with corresponding sentence.
        max_prob, max_sent = 0.0, None

        for i, sent in enumerate(sentences):
            prob = 1 
            for w in data_list[i] : 
                prob *= probs[w] 
            
            if max_prob < prob:
                max_prob, max_sent = prob, sent

        if len(max_sent) > 0 :
            summary.append(max_sent)

        sentences.remove(max_sent)

        if update:
            # Apply the update for non-redundancy.
            for w in data_list[i]:
                probs[w] = probs[w] ** 2
  
    return summary 

summary_sumbasic = sumbasic_summarize(LIMIT, data['clean_text'],update=False)

file_sum_basic = open('Data/' + DATASET + '/SumBasic.txt', 'w+')

for sentence in summary_sumbasic :
    row_no = data[data['clean_text'] == str(sentence)].index[0]
    file_sum_basic.write(data.iloc[row_no]['text'] + "\n") 

file_sum_basic.close()
        
