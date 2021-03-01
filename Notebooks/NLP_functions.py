import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize, pos_tag
from nltk import bigrams
import re
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Init Lemmatizer, stemmer, and vader sentiment analysis
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
analyzer = SentimentIntensityAnalyzer()

def display_topics(model, feature_names, no_top_words, topic_names):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def prob_df(X, topics):
    '''
    df of all documents as rows and the probablities of each topic as columns
        
    input: 
    X = model X components,
    topics = list of topic names,
        
    output: df 
    '''
    cols = topics

    df = pd.DataFrame(X.round(5), columns = cols)
    df['topic_choice'] = df.idxmax(axis=1)
    df['topic_value'] = df.max(axis=1)

    return df

def topic_matrix(model, feature_names, topics):
    '''
    df of all words as rows and the probablities of each topic as columns
        
    input: 
    model = model,
    feature_names = words in bag of words,
    topics = list of topic names,
        
    output: df 
    '''
    idx = topics

    topic_word = pd.DataFrame(model.components_.round(3),
             index = idx,
             columns = feature_names)
    
    topic_word = topic_word.T
    topic_word['topic_choice'] = topic_word.idxmax(axis=1)
    topic_word['max_value'] = topic_word.max(axis=1)
    
    return topic_word

def clean_str(comment, stop, replace):
        '''
        clean comment string, removing numbers, punctuation, non alphabetical words, replacing words,
        stop words, and changing to lowercase, 
        lemmatizing and stemming
        
        input: 
        comment = comment to be cleaned,
        stop = stopwords,
        replace = dictionary of words to replace
        
        output: cleaned string
        '''

        s = str(comment.lower()) #lowercase
            
        s = re.sub(r'\([^)]*\)', '', s) #remove links
            
        #only alphabetical letters
        NON_ASCII = re.compile(r'[^a-z0-1\s]')
        s = NON_ASCII.sub(r'', s)
        
        s = re.sub(r'[0-9]+', '', s) #remove numbers
            
        #replace acronyms/names 
        for key, value in replace.items():
            s = s.replace(key, value) 
    
        s_token = word_tokenize(s) #tokenize
        
        s_token = [w.strip() for w in s_token] #remove spaces
        
        s_token = [w for w in s_token if w not in stop] #remove stop words
            
        s_token_lemm = [lemmatizer.lemmatize(w) for w in s_token] #Lemmatize
        
        s_token_lemm_stem = [stemmer.stem(w) for w in s_token_lemm] #stem
            
        s_token_lemm_stem = [w for w in s_token_lemm_stem if w not in stop] #remove stop words
        
        s_lemm_stem = " ".join(s_token_lemm_stem) #join back into one string
        
        return s_lemm_stem

def pos_only(row):
    '''
    find all nouns verbs and adjectives in a string
    
    input: row of df
    
    output: string of all nouns verbs and adjectives
    '''
    s = row['cleaned_comments']
    
    s_token = word_tokenize(s)  # tokenize words
    
    s_pos = pos_tag(s_token) #find parts of speech
    
    pos = []
    for x in s_pos:
        if x[1] == 'NN' or x[1] == 'VB' or x[1] == 'JJ':
            pos.append(x[0])
    
    return ' '.join([word for word in pos])

def get_top_n_words(corpus, stop, n=10):
    '''
    get top n words and frequencies
    
    input: 
    corpus = array of strings to count
    n = number of top words to return
    
    output: sorted dictionary of top words and frequencies
    '''
    vec = CountVectorizer(stop_words=stop).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) #find frequency

    words_freq = [(word, sum_words[0, idx]) for word, idx in   vec.vocabulary_.items()] #create dictionary and sort
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def find_top_bigrams( corpus):
    '''
    get top bigrams frequencies
    
    input: corpus = array of strings to count
    
    output: sorted dictionary of top bigrams and frequencies
    '''
    bigrams_list = list(bigrams(corpus))

    dictionary2 = [' '.join(tup) for tup in bigrams_list]

    #Using count vectoriser to view the frequency of bigrams
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    bag_of_words = vectorizer.fit_transform(dictionary2)
    vectorizer.vocabulary_
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return dict(words_freq)

def top_topics_df(model, feature_names, no_top_words, topic_names, replace):
    '''
    df of all top words for each topic 
        
    input: 
    model = model,
    feature_names = words in bag of words,
    no_top_words = number of top words to include,
    topic_names = list of topic names,
    replace = dictionary of words to replace
        
    output: df 
    '''
    
    df = pd.DataFrame()
    for ix, topic in enumerate(model.components_):
        words = topic_names[ix] + '_words'
        
        l = topic.argsort()[:-no_top_words - 1 :-1]

        l_word = []
        for x in l:
            w = feature_names[x]
            #replace words 
            for key, value in replace.items():
                w = w.replace(key, value) 

            l_word.append(w.upper())

        df[words] = l_word

        value = topic_names[ix] + '_values'
        df[value] = np.sort(topic)[:-no_top_words - 1 :-1]
    df.drop(df.index[0], inplace=True)

    return df

def get_sentiments(df):
    """
    add sentiment analysis to df
    """
    df['compound'] = df['comment'].apply(lambda comment: analyzer.polarity_scores(str(comment))['compound'])
    df['positive'] = df['comment'].apply(lambda comment: analyzer.polarity_scores(str(comment))['pos'])
    df['negative'] = df['comment'].apply(lambda comment: analyzer.polarity_scores(str(comment))['neg'])
    df['sentiment'] = df['compound'].apply(lambda c: 'posititve' if c >=0 else 'negative')

    return df