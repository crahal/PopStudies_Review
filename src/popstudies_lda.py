import re
import os
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from tqdm.notebook import tqdm
# import guidedlda
from topics import seed_topic_list
from warnings import simplefilter
import logging
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models, matutils
from gensim.models.wrappers import LdaMallet
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
logger = logging.getLogger()
logging.disable(logging.CRITICAL)


def print_lda_output_topics(d_path):
    with open(os.path.join(d_path, 'models',
                           'lda_words_by_topic.txt'), 'r') as f:
        lineArr=f.read().split('\n')
        for line in lineArr:
            print(line)


def make_mallet_model(main_df, d_path, stop, field, ntopics):
    mallet_path = 'mallet/bin/mallet'
    main_df_notnull = main_df[main_df['abstract'].str.strip()!='nan.'].copy()
    main_df_notnull = main_df_notnull[main_df_notnull['abstract_length']>20]
    main_df_notnull = main_df_notnull[main_df_notnull['Title'].notnull()]
    token_vectorizer = CountVectorizer(tokenizer=reflection_tokenizer,
                                       #max_df=500, min_df=2,
                                       stop_words=stop, ngram_range=(1, 3))
    token_vectorizer.fit(main_df_notnull[field])
    doc_word = token_vectorizer.transform(main_df_notnull[field]).transpose()
    corpus = matutils.Sparse2Corpus(doc_word)
    word2id = dict((v, k) for v, k in token_vectorizer.vocabulary_.items())
    id2word = dict((v, k) for k, v in token_vectorizer.vocabulary_.items())
    dictionary = corpora.Dictionary()
    dictionary.id2token = id2word
    dictionary.token2id = word2id
    texts = main_df_notnull[field].apply(lambda x: x.split()).to_list()
    ldamallet = LdaMallet(mallet_path, corpus=corpus, num_topics=ntopics,
                          id2word=id2word, random_seed = 77)
    mallet_topics = pd.DataFrame(index=list(id2word.values()),
                                 columns = ['Topic ' + str(x) for x in range(1, ntopics+1)])
    print(ldamallet.show_topics(num_topics=ntopics,
                                num_words=10, formatted=True))
    for topic in ldamallet.show_topics(num_topics=ntopics,
                                       num_words=len(id2word), formatted=False):
        for tupler in topic[1]:
            mallet_topics.loc[tupler[0], 'Topic ' + str(topic[0]+1)] = tupler[1]
    mallet_topics.to_csv(os.path.join(d_path, 'models', 'mallet_topic_df.csv'))
    return mallet_topics


def make_lda_model(main_df, d_path, stop):
    main_df_notnull = main_df[main_df['clean_abstract'].notnull()].copy()
    main_df_notnull = main_df_notnull[main_df_notnull['abstract_length']>20]
    ngram_range = range(1, 2, 1)
    conf_upper = 90
    conf_range = range(50, conf_upper, 1)
    confidence_df = pd.DataFrame(index=conf_range, columns=ngram_range)
    for ngram_limit in ngram_range:
        print('working on ngram_limit: ' + str(ngram_limit))
        token_vectorizer = CountVectorizer(tokenizer=reflection_tokenizer,
                                           max_df=500, stop_words=stop,
                                           min_df=2,
                                           ngram_range=(1, ngram_limit))
        X_ngrams = token_vectorizer.fit_transform(main_df_notnull['clean_abstract'])
        tf_feature_names = token_vectorizer.get_feature_names()
        word2id = dict((v, idx) for idx, v in enumerate(tf_feature_names))
        model = guidedlda.GuidedLDA(n_topics=len(seed_topic_list),
                                    n_iter=10000, #potentially make this large
                                    random_state=7, refresh=10)
        seed_topics = {}
        for t_id, st in enumerate(seed_topic_list):
            for word in st:
                try:
                    seed_topics[word2id[word]] = t_id
                except KeyError:
                    print('Problem seeding: ' + word + '!')
        for conf in tqdm(conf_range):
            conf = conf/conf_upper
            model.fit(X_ngrams, seed_topics=seed_topics, seed_confidence=conf)
            confidence_df.loc[conf*conf_upper, ngram_limit] = model.loglikelihood()
    v = confidence_df.values
    i, j = [x[0] for x in np.unravel_index([np.argmax(v)], v.shape)]
    token_vectorizer = CountVectorizer(tokenizer = reflection_tokenizer,
                                       min_df=10, stop_words=stop,
                                       ngram_range=(1,confidence_df.columns[j]))
    X_ngrams = token_vectorizer.fit_transform(main_df_notnull['clean_abstract'])
    tf_feature_names = token_vectorizer.get_feature_names()
    word2id = dict((v, idx) for idx, v in enumerate(tf_feature_names))
    model.fit(X_ngrams, seed_topics=seed_topics, seed_confidence=confidence_df.index[i]) #pickle this
    n_top_words = 20 #Print out the ten words most associate with each topic
    topic_word = model.topic_word_
    with open(os.path.join(d_path, 'models', 'lda_words_by_topic.txt'), "w") as f:
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(tf_feature_names)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)), file=f)
    doc_topic = model.transform(X_ngrams)
    columns_label = ['topic {}'.format(i) for i in range(len(seed_topic_list))]  # number of topics
    topic_vector = pd.DataFrame(doc_topic, columns = columns_label)#dataframe of doc-topics
    topic_vector.to_csv(os.path.join(d_path, 'models', 'topic_vector.csv'))
    confidence_df.to_csv(os.path.join(d_path, 'models', 'conf_dict.csv'))


def get_wordnet_pos(word):
    '''tags parts of speech to tokens
    Expects a string and outputs the string and
    its part of speech'''

    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def word_lemmatizer(text):
    '''lemamtizes the tokens based on their part of speech'''
    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(text, get_wordnet_pos(text))
    return text


def reflection_tokenizer(text):
    stop_list = pd.read_csv(os.path.join(os.getcwd(), '..', 'data', 'support',
                                         'custom_stopwords.txt'))
    custom_stop = stop_list['words'].to_list()
    stop = nltk.corpus.stopwords.words('english')
    for word in custom_stop:
        stop.append(word)

    text = re.sub(r'[\W_]+', ' ', text)  # keeps alphanumeric characters
    text = re.sub(r'\d+', '', text)  # removes numbers
    text = text.lower()
    tokens = [word for word in word_tokenize(text)]
    tokens = [word for word in tokens if len(word) >= 3]
    # removes smaller than 3 character
    tokens = [word_lemmatizer(w) for w in tokens]
    tokens = [s for s in tokens if s not in stop]
    return tokens
