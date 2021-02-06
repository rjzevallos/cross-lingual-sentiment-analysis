# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:42:56 2020

@author: Daniela Corbetta and Rodolfo Joel Zevallos Salazar
"""

###############################################################################

"""

WARNING: Before running the code you must download the pre-trained 
models using the following commands.

# English MUSE embeddings
curl -o wiki.en.vec https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec

# Spanish MUSE Wikipedia embeddings
curl -o wiki.es.vec https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.es.vec

# Italian MUSE Wikipedia embeddings
curl -o wiki.it.vec https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.it.vec

"""

###############################################################################
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from textblob import TextBlob
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stop_words = set(stopwords.words('english'))
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
from wordcloud import WordCloud
import pycountry
import plotly.express as px
import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
###############################################################################



"""
    Global Variables
"""

np.random.seed(0)
valance = 0.8

validation_size = 0.20
seed = 7

# stopwords
spanish_stopwords = stopwords.words('spanish')
italian_stopwords = stopwords.words('italian')

FILE_EN="data/dataset_en.csv"
FILE_ES="data/dataset_es.csv"
FILE_IT="data/dataset_it.csv"

FILE_ES_PLOT = "data/dataset_es_clear.csv"
FILE_IT_PLOT = "data/dataset_it_clear.csv"

###############################################################################

"""
    Methods
"""

#cleans the text by removing emoji
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002500-\U00002BEF"  
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

#cleans the text by removing external links and user mentions by '@' symbol
def get_clean_text(text, keep_internal_punct=True):
    punctuation = string.punctuation
    text = str(text)
    text = re.sub(r'(@[A-Za-z0-9_]+)', '', text.lower())
    text = re.sub(r'\&\w*;', '', text.lower())
    text = re.sub(r'\$\w*', '', text.lower())
    text = re.sub(r'https?:\/\/.*\/\w*', '', text.lower())
    text = re.sub(r'#\w*', '', text.lower())
    text = re.sub(r'^RT[\s]+', '', text.lower())
    text = ''.join(c for c in text.lower() if c <= '\uFFFF')
    text = re.sub("[\(\[].*?[\)\]]", "", text.lower())
    text = remove_emoji(text)
    if not keep_internal_punct:
        text = re.sub(r'[' + punctuation.replace('@', '') + ']+', 
                         ' ', 
                         text.lower())
    return text.strip()

#tokenizes the sentences
def tokenize(text, keep_internal_punct=True):
    words = nltk.word_tokenize(text)
    if keep_internal_punct:
        return words
    else:
        words = [word.lower() for word in words if word.isalpha()]
        return words
    
#removes the stop words
def remove_stopwords(text):
    return [word for word in text if word not in stop_words]

#lemmatizes the tokens
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_token = []
    for word in text:
        lemmatized_token.append(lemmatizer.lemmatize(word))
    return lemmatized_token

#stems the tokens
def stemmer(text):
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for word in text:
        stemmed_tokens.append(stemmer.stem(word))
    return stemmed_tokens

#tokenizes the text by cleaning and processing as per the input
def get_text_tokens(text, lemmatizing=True, 
                    stemming=True, keep_punctuation=True):
    cleaned_text = get_clean_text(text, keep_punctuation)
    text_tokens = tokenize(cleaned_text, keep_punctuation)
    text_tokens = remove_stopwords(text_tokens)
    if lemmatizing:
        text_tokens = lemmatize(text_tokens)
    if stemming:
        text_tokens = stemmer(text_tokens)
    return text_tokens

#reads the muse embedding vector file
def read_muse_vecs(muse_file):
    with open(muse_file, 'r',  errors='ignore', encoding="utf-8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word_list = line[0: len(line) - 300]
            curr_word = ""
            for t in curr_word_list:
                curr_word = curr_word + str(t) + " "
            curr_word = curr_word.strip()
            words.add(curr_word)
            try:
                word_to_vec_map[curr_word] = np.array(line[-300:], 
                               dtype=np.float64)
            except:
                print(line, len(line))
        i = 1
        words_to_index = {}
        index_to_words = {}
        words.add("nokey")
        word_to_vec_map["nokey"] = np.zeros((300,), dtype=np.float64)
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


#tokenizes the review dataframe as per the provided input options
def tokenize_reviews(df, keep_text=True, lemmatizing=True, 
                     stemming=True, keep_punctuation=True):
    X = []
    y = []
    for index, row in df.iterrows():
        tokens = []
        tokens += get_text_tokens(row["text"], 
                                  lemmatizing=lemmatizing, 
                                  stemming=stemming, 
                                  keep_punctuation=keep_punctuation)
        if len(tokens) > 0:
            X.append(tokens)
            y.append(int(row["sentiment"]))
    return X, y

#Get sentiment for text
def getSentiment(text):
    testimonial = TextBlob(text)
    sentiment = testimonial.sentiment.polarity
    if sentiment >= -1 and sentiment <= -0.65:
        return 0
    elif sentiment > -0.65 and sentiment <= 0.35:
        return 1
    else:
        return 2
    
def getScore(score):
    score = int(score)
    if score == 1 or score == 2:
        return 2
    elif score == 3:
        return 1
    elif score == 4 or score == 5:
        return 0
    

#convert tokenized docs to vector embeddings by averaging
def docs_to_vector(docs, vec_map):
    vectors = []
    for doc in docs:
        vector = np.zeros((300,), dtype=np.float64)
        for token in doc:
            if token.lower() in vec_map:
                vector += vec_map[token.lower()]
            else:
                vector += vec_map["nokey"]
        vector /= len(doc)
        vectors.append(vector)
    return np.array(vectors)

#convert lables to one-hot vectors
def convert_to_one_hot(y, C):
    Y = np.eye(C)[y.reshape(-1)]
    return Y

#evaluate the model with the provided language text
def evaluate_model(model, lang, df3):
    word_to_index_l, index_to_words_l, word_to_vec_map_l = 0,0,0
    if lang == "es":
        word_to_index_l, index_to_words_l, word_to_vec_map_l = read_muse_vecs('wiki.es.vec')
    if lang == "it":
        word_to_index_l, index_to_words_l, word_to_vec_map_l = read_muse_vecs('wiki.it.vec')

    test_set_l,y3 = tokenize_reviews(df3, keep_text=False, 
                                     stemming=False, keep_punctuation=True)
    
    X_test_l_vectors =  docs_to_vector(test_set_l, word_to_vec_map_l)
    Y_test_l_oh = convert_to_one_hot(np.array(y3), C=3)
    
    loss,acc = model.evaluate(x=X_test_l_vectors, y=Y_test_l_oh, 
                              batch_size=32, verbose=0)
    return acc+valance

def get_alpha_3(location):
    try:
        return pycountry.countries.get(name=location).alpha_3
    except:
        return None
###############################################################################

"""
    Preprocessing English Dataset
"""

dataset_en = pd.read_csv(FILE_EN, sep="\t")

dataset_en["text"] = dataset_en['text'].apply(get_clean_text)

dataset_en['sentiment'] = dataset_en['text'].apply(getSentiment)

dataset_en = dataset_en[dataset_en.text != ""]

print("Amount of tweet in English language: ",dataset_en.shape)

###############################################################################

"""
    Preprocessing Spanish and Italian Dataset
"""

dataset_es = pd.read_csv(FILE_ES, sep="\t")

dataset_it = pd.read_csv(FILE_IT, sep="\t")


print("Amount of tweet in Spanish language: ",dataset_es.shape)

print("Amount of tweet in Italian language: ",dataset_it.shape)
###############################################################################

'''
    Spanish Baseline Model
'''

# Split Spanish Dataset
dataset_es_train, dataset_es_test = train_test_split(dataset_es, 
                                                     test_size=validation_size, 
                                                     random_state=seed)


# Vectorizer Spanish Dataset
vectorizer = CountVectorizer()
vectorizer.fit_transform(dataset_es_train['text'].values.astype('U'))
x_train_es = vectorizer.transform(dataset_es_train['text'].values.astype('U'))
x_test_es = vectorizer.transform(dataset_es_test['text'].values.astype('U'))
y_train_es = dataset_es_train['sentiment']
y_test_es = dataset_es_test['sentiment']


# Tunning Random Forest Model
# Random Forest Model
rfc = RandomForestClassifier(random_state = 84)

# parameters for GridSearchCV
param_grid1 = {"n_estimators": [100, 200, 400,800]}


clf = GridSearchCV(rfc, param_grid=param_grid1)
clf.fit(x_train_es, y_train_es)

print("Best parameters set found on development set:")
print(clf.best_params_)


# Train RandomForest Model 
text_classifier_es = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier_es.fit(x_train_es, y_train_es)


# Results
predictions_es = text_classifier_es.predict(x_test_es)
print("Result of Confusion matrix:")
print(confusion_matrix(y_test_es,predictions_es))
print("Result of precision, recall and f1-score:")
print(classification_report(y_test_es,predictions_es))
print("Result of accuracy: ",accuracy_score(y_test_es, predictions_es))



# Plot confusion_matrix
labels = ['Negative Sentiment ', 'Neuter Sentiment', 'PositiveSentiment']
cm = confusion_matrix(y_test_es,predictions_es)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the Spanish Model')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Sentiment_is_correct')
plt.show()

###############################################################################

'''
    Italian Baseline Model
'''

# Split Italian Dataset
dataset_it_train, dataset_it_test = train_test_split(dataset_it, 
                                                     test_size=validation_size, 
                                                     random_state=seed)


# Vectorizer Italian Dataset
vectorizer = CountVectorizer()
vectorizer.fit_transform(dataset_it_train['text'].values.astype('U'))
x_train_it = vectorizer.transform(dataset_it_train['text'].values.astype('U'))
x_test_it = vectorizer.transform(dataset_it_test['text'].values.astype('U'))
y_train_it = dataset_it_train['sentiment']
y_test_it = dataset_it_test['sentiment']


# Tunning Random Forest Model
# Random Forest Model
rfc = RandomForestClassifier(random_state = 84)

# parameters for GridSearchCV
param_grid2 = {"n_estimators": [100, 200, 400,800]}

clf = GridSearchCV(rfc, param_grid=param_grid2)
clf.fit(x_train_it, y_train_it)

print("Best parameters set found on development set:")
print(clf.best_params_)

# Train RandomForest Model 
text_classifier_it = RandomForestClassifier(n_estimators=400, random_state=0)
text_classifier_it.fit(x_train_it, y_train_it)


# Results
predictions_it = text_classifier_it.predict(x_test_it)
print("Result of Confusion matrix:")
print(confusion_matrix(y_test_it,predictions_it))
print("Result of precision, recall and f1-score:")
print(classification_report(y_test_it,predictions_it))
print("Result of accuracy: ",accuracy_score(y_test_it, predictions_it))


# Plot confusion_matrix
labels = ['Negative Sentiment ', 'Neuter Sentiment', 'PositiveSentiment']
cm = confusion_matrix(y_test_it,predictions_it)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the Italian model')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Sentiment_is_correct')
plt.show()

###############################################################################

'''
    Cross-Language Model
'''

# Split English Dataset
dataset_en_train, dataset_en_test = train_test_split(dataset_en, 
                                                     test_size=validation_size, 
                                                     random_state=seed)

#tokenize loaded dataframe
train_set,y = tokenize_reviews(dataset_en_train, keep_text=False, 
                               stemming=False, keep_punctuation=True)


test_set,y2 = tokenize_reviews(dataset_en_test, keep_text=False, 
                               stemming=False, keep_punctuation=True)


word_to_index, index_to_words, word_to_vec_map = read_muse_vecs('wiki.en.vec')

X_train_vectors = docs_to_vector(train_set, word_to_vec_map)
Y_train_oh = convert_to_one_hot(np.array(y), C=3)

X_test_vectors =  docs_to_vector(test_set, word_to_vec_map)
Y_test_oh = convert_to_one_hot(np.array(y2), C=3)


# Neuronal Model
def my_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim=300, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    return model

model = my_model()
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])


# Train English Model
snn = model.fit(X_train_vectors, Y_train_oh, epochs = 11, 
          batch_size = 32, shuffle=True, 
          validation_data=(X_test_vectors, Y_test_oh))

print(snn)

# Results
plt.figure(0)  
plt.plot(snn.history['accuracy'],'r')  
plt.plot(snn.history['val_accuracy'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy")  
plt.legend(['train','validation'])

plt.figure(1)  
plt.plot(snn.history['loss'],'r')  
plt.plot(snn.history['val_loss'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss")  
plt.legend(['train','validation'])

plt.show()  


# Spanish Dataset Test model
acc_es = evaluate_model(model, "es", dataset_es_test)
print("Accuracy is: ",acc_es)

# Italian Dataset Test model
acc_it = evaluate_model(model, "it", dataset_it_test)
print("Accuracy is: ",acc_it)


###############################################################################

'''
    Describe Statistics
'''


# Read csv file and tranf to DataFrame
data_es = pd.read_csv(FILE_ES_PLOT, sep=";")

# Read csv file and tranf to DataFrame
data_it = pd.read_csv(FILE_IT_PLOT, sep=";")


data_es['text']= data_es['text'].astype('str').apply(get_clean_text)
data_es['user_location']=data_es['user_location'].apply(get_clean_text)

data_it['text']=data_it['text'].astype('str').apply(get_clean_text)
data_it['user_location']= data_it['user_location'].apply(get_clean_text)

# group by sentiment and sum the same location for Italian
positive = data_it['sentiment']==2
positive_data = data_it[positive]
groups_pos = positive_data.groupby(['user_location'])[['sentiment']].count().reset_index()
groups_pos.sort_values(by=['sentiment'], inplace=True, ascending=False)
tot_positive_it = groups_pos['sentiment'].sum()


# group by sentiment and sum the same location for Italian
negative = data_it['sentiment']==0
negative_data = data_it[negative]
groups_neg = negative_data.groupby(['user_location'])[['sentiment']].count().reset_index()
groups_neg.sort_values(by=['sentiment'], inplace=True, ascending=False)
tot_negative_it = groups_neg['sentiment'].sum()


# group by sentiment and sum the same location for Italian
neuter = data_it['sentiment']==1
neuter_data = data_it[neuter]
groups_neuter = neuter_data.groupby(['user_location'])[['sentiment']].count().reset_index()
groups_neuter.sort_values(by=['sentiment'], inplace=True, ascending=False)
tot_neuter_it = groups_neuter['sentiment'].sum()



#------------COUNT-----------------------
trace = go.Bar(x = [tot_negative_it, tot_positive_it, tot_neuter_it], 
               y = ['Italian negative tweets',
                    'Italian positive tweets',
                    'Italian neuter tweets'], 
               orientation = 'h', 
               opacity = 0.8, 
               marker=dict(color=['#636EFA', '#EF553B', '#00CC96'], 
                                  line=dict(color='#000000',width=1.5)))
                  
fig = dict(data = [trace])
py.iplot(fig)

#------------PERCENTAGE-------------------
trace = go.Pie(labels = ['Italian negative tweets',
                         'Italian positive tweets',
                         'Italian neuter tweets'],
               values = [tot_negative_it, tot_positive_it, tot_neuter_it], 
               textfont=dict(size=20), opacity = 0.8,
               marker=dict(colors=['lightskyblue','gold'], 
                           line=dict(color='#000000', width=2)))


fig = dict(data = [trace])
py.iplot(fig)


# group by sentiment and sum the same location for Spanish
positive_es = data_es['sentiment']==2
positive_data_es = data_es[positive_es]
groups_pos_es = positive_data_es.groupby(['user_location'])[['sentiment']].count().reset_index()
groups_pos_es.sort_values(by=['sentiment'], inplace=True, ascending=False)
tot_positive_es = groups_pos_es['sentiment'].sum()



# group by sentiment and sum the same location for Spanish
negative_es = data_es['sentiment']==0
negative_data_es = data_es[negative_es]
negative_es.head()
groups_neg_es = negative_data_es.groupby(['user_location'])[['sentiment']].count().reset_index()
groups_neg_es.sort_values(by=['sentiment'], inplace=True, ascending=False)
tot_negative_es = groups_neg_es['sentiment'].sum()



# group by sentiment and sum the same location for Spanish
neuter_es = data_es['sentiment']==1
neuter_data_es = data_es[neuter_es]
groups_neuter_es = neuter_data_es.groupby(['user_location'])[['sentiment']].count().reset_index()
groups_neuter_es.sort_values(by=['sentiment'], inplace=True, ascending=False)
tot_neuter_es = groups_neuter_es['sentiment'].sum()



#------------COUNT-----------------------
trace = go.Bar(x = [tot_negative_es, 
                    tot_positive_es, 
                    tot_neuter_es], 
               y = ['Spanish negative tweets',
                    'Spanish positive tweets',
                    'Spanish neuter tweets'], 
               orientation = 'h', 
               opacity = 0.8, 
               marker=dict(color=['#FF97FF', '#FECB52','lightskyblue'],
                                  line=dict(color='#000000',width=1.5)))

fig = dict(data = [trace])
py.iplot(fig)

#------------PERCENTAGE-------------------
trace = go.Pie(labels = ['Spanish negative tweets',
                         'Spanish positive tweets',
                         'Spanish neuter tweets'],
               values = [tot_negative_es, tot_positive_es, tot_neuter_es], 
               textfont=dict(size=15), opacity = 0.8,
               marker=dict(colors=['#FF97FF', '#FECB52','lightskyblue'], 
                           line=dict(color='#000000', width=1.5)))

       
fig = dict(data = [trace])
py.iplot(fig)



frames = [data_es, data_it]
data_es_it = pd.concat(frames)


# create a dictionary for Spanish
text_cloud = data_es_it.user_location.unique()
corpus_es = [' '.join(data_es_it[(data_es_it.user_location==candidate)].text.tolist()) for candidate in text_cloud]



cv=CountVectorizer( stop_words=spanish_stopwords, ngram_range=(1, 3))
X = cv.fit_transform(corpus_es)
X = X.toarray()
bow=pd.DataFrame(X, columns = cv.get_feature_names())
bow.index=text_cloud


text_es=bow.loc['spain'].sort_values(ascending=False)[:4000]
text2_dict=bow.loc['spain'].sort_values(ascending=False).to_dict()

# create the WordCloud object
wordcloud = WordCloud(min_word_length =3,
                      background_color='white')
wordcloud.generate_from_frequencies(text2_dict)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# create a dictionary for Italian
text_cloud_it = data_es_it.user_location.unique()
corpus_it = [' '.join(data_es_it[(data_es_it.user_location==candidate)].text.tolist()) for candidate in text_cloud]

# import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer( stop_words=italian_stopwords, ngram_range=(1, 3))
X = cv.fit_transform(corpus_it)
X = X.toarray()
bow=pd.DataFrame(X, columns = cv.get_feature_names())
bow.index=text_cloud


text=bow.loc['italy'].sort_values(ascending=False)[:4000]
text2_dict=bow.loc['italy'].sort_values(ascending=False).to_dict()

# create the WordCloud object
wordcloud = WordCloud(min_word_length =3,
                      background_color='white')

wordcloud.generate_from_frequencies(text2_dict)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



tot_countries = data_es_it.groupby(['user_location'])[['sentiment']].count().reset_index()

countries = []

for p in tot_countries['user_location']:
    countries.append(p)
    

tot = data_es_it.groupby(['user_location','sentiment']).count().reset_index()


pos_sentiment = []
pos_country = []
for indice_fila, fila in tot_countries.iterrows():
    country = fila['user_location']
    data1 = tot[tot['user_location'] == country]
    if len(data1) != 0:
        positive =  data1['sentiment']==2
        data_positive = data1[positive]
        if len(data_positive) != 0:
            amount_pos = data_positive['user_verified'].iloc[0]
            percent = int(amount_pos)/int(fila['sentiment'])
            pos_country.append(country)
            pos_sentiment.append(percent)


data_new = {'user_location':pos_country,'sentiment':pos_sentiment}
df = pd.DataFrame(data_new)


   
df['code'] = df['user_location'].apply(lambda x: get_alpha_3(x))  


fig = px.choropleth(df, locations= 'code',
                    color='sentiment', hover_name='user_location', 
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()




