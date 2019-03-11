# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import gc
from gensim.models import KeyedVectors
import re

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
print("train shape: ", df_train.shape)
print("test shape: ", df_test.shape)

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
          '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
          '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
          'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
          '≤', '‡', '√', ]

mispell_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
                "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
                "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would",
                "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have",
                "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
                "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center',
                'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater',
                'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2',
                'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
                'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best',
                'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate',
                "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist',
                'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
                'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                'demonitization': 'demonetization', 'demonetisation': 'demonetization'}


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispellings, mispellings_re = _get_mispell(mispell_dict)


def clean_text(x):
    x = str(x)
    x = x.lower()
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)


    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, x)


print('Preprocessing data......')
df_train.question_text = df_train.question_text.apply(lambda x: clean_text(x))
df_test.question_text = df_test.question_text.apply(lambda x: clean_text(x))

df_train.question_text = df_train.question_text.fillna('_##_')
df_test.question_text = df_test.question_text.fillna('_##_')

tokenizer = Tokenizer()
data = list(df_train.question_text.values) + list(df_test.question_text.values)
tokenizer.fit_on_texts(data)

train_data = tokenizer.texts_to_sequences(df_train.question_text.values)
train_label = df_train.target.values
train_data = pad_sequences(train_data, maxlen=100)
X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.05, random_state=40)

test_data = tokenizer.texts_to_sequences(df_test.question_text.values)
test_data = pad_sequences(test_data, maxlen=100)

embed_size = 300
max_features = 95000


def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    print('---load %s' % EMBEDDING_FILE)

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    del embeddings_index, embedding_vector
    gc.collect()

    return embedding_matrix


def load_fasttext(word_index):
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    print('---load %s' % EMBEDDING_FILE)

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    del embeddings_index, embedding_vector
    gc.collect()

    return embedding_matrix


def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    print('---load %s' % EMBEDDING_FILE)

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    del embeddings_index, embedding_vector
    gc.collect()

    return embedding_matrix


# def load_googlenews(word_index):
#     EMBEDDING_FILE = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
#     print('---load %s'%EMBEDDING_FILE)
#     embeddings_index = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

#     nb_words = min(max_features, len(word_index))
#     embedding_matrix = (np.random.rand(nb_words, embed_size) - 0.5) / 5.0
#     for word, i in word_index.items():
#         if i >= max_features: continue
#         if word in embeddings_index:
#             embedding_vector = embeddings_index.get_vector(word)
#             embedding_matrix[i] = embedding_vector

#     del embeddings_index, embedding_vector
#     gc.collect()

#     return embedding_matrix


embedding_matrix_glove = load_glove(tokenizer.word_index)
embedding_matrix_fasttext = load_fasttext(tokenizer.word_index)
embedding_matrix_para = load_para(tokenizer.word_index)
# embedding_matrix_googlenews = load_googlenews(tokenizer.word_index)

print('------hybrid embedding matrix------')
embedding_matrix = np.concatenate((embedding_matrix_glove, embedding_matrix_fasttext, embedding_matrix_para), axis=1)
del embedding_matrix_glove, embedding_matrix_fasttext, embedding_matrix_para
gc.collect()
print('embedding matrix shape: {}'.format(embedding_matrix.shape))


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, axis=2)
    if time_major:
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])
    print('inputs shape: {}'.format(inputs.shape))
    hidden_size = inputs.shape[2].value

    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (Batch, Time)
    alphas = tf.nn.softmax(vu, name='alphas')  # (Batch, Time)

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)  # (Batch, Dimension)
    if not return_alphas:
        return output
    else:
        return output, alphas


TIME_STEPS = 100

# HIDDEN_UNITS1 = 150
# HIDDEN_UNITS = 150
LEARNING_RATE = 0.01
ATTENTION_SIZE = 50
HIDDEN_SIZE = 100

with tf.name_scope('Model'):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        global_step = tf.Variable(0, trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=global_step)
        ema_op = variable_averages.apply(tf.trainable_variables())
        X = tf.placeholder(dtype=tf.int32, shape=(None, TIME_STEPS), name="input_placeholder")
        Y_pred = tf.placeholder(dtype=tf.float32, shape=(None), name='pred_placeholder')
        keep_prob = tf.placeholder(dtype=tf.float32, shape=(None), name='keep_prob')
        # threshold = tf.Variable(tf.random_uniform([1],0.0,1.0,dtype=tf.float32)[0],trainable=True)
        word_emb = tf.get_variable('word_emb', initializer=tf.constant(embedding_matrix, dtype=tf.float32),
                                   trainable=False)
        X1 = tf.nn.embedding_lookup(word_emb, X)
        print('X1 shape: ', X1.shape)
        # ------------- bidirectional LSTM-------------------
        lstm_forward = tf.nn.rnn_cell.LSTMCell(num_units=HIDDEN_SIZE, name="forward_lstm_0")
        lstm_backward = tf.nn.rnn_cell.LSTMCell(num_units=HIDDEN_SIZE, name="backward_lstm_0")
        inputs_0, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_forward, cell_bw=lstm_backward, inputs=X1,
                                                      dtype=tf.float32)
        # inputs_1 = tf.concat([inputs_0[0],inputs_0[1]],axis=2)

        attention_output, alphas = attention(inputs_0, ATTENTION_SIZE, return_alphas=True)
        attention_input = tf.nn.dropout(attention_output, keep_prob=keep_prob)
        print('attention_input shape: {}'.format(attention_input.shape))
    with tf.name_scope('fully_connect_layer'):
        w = tf.Variable(tf.truncated_normal([HIDDEN_SIZE * 2, 1], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[1]))
        outputs = tf.nn.xw_plus_b(attention_input, w, b)
        print("outputs shape: ", outputs.shape)
        outputs = tf.squeeze(outputs)
    with tf.name_scope('Metrics'):
        result = tf.nn.sigmoid(outputs)
        # accuracy = accuracy_score(Y_pred, (result0>0.33).astype(int))
        # learning_rate = tf.train.exponential_decay(0.01,global_step=global_step,decay_steps=2000,decay_rate=0.5,staircase=True)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_pred, logits=outputs))
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss=loss, global_step=global_step)

        train_op = tf.group([optimizer, ema_op])


# Batch generators
def batch_generator(X, y, batch_size):
    size = X.shape[0]
    indices = np.arange(size)
    np.random.shuffle(indices)
    i = 0
    while True:
        if i + batch_size <= size:
            yield X[i:i + batch_size], y[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            continue


NUM_EPOCHS = 2
DELTA = 0.5
BATCH_SIZE = 128
train_batch_generator = batch_generator(X_train, y_train, BATCH_SIZE)
val_batch_generator = batch_generator(X_val, y_val, BATCH_SIZE)

sess = tf.Session()
initializer = tf.initializers.global_variables()
sess.run(initializer)

for epoch in range(NUM_EPOCHS):
    print("epoch: {}\t".format(epoch), end="")
    # loss_train = 0
    # loss_test = 0
    # acc_train = 0
    # acc_val = 0
    # Training
    for i in range(9694):
        x_batch, y_batch = next(train_batch_generator)
        _, train_loss, result0 = sess.run([train_op, loss, result],
                                          feed_dict={X: x_batch, Y_pred: y_batch, keep_prob: 0.8})
        # loss_train = loss_train*(1-DELTA) + train_loss * DELTA
        # train_acc = accuracy_score(y_batch, (result0>0.33).astype(int))
        # acc_train += train_acc
        # # Validation
        # if i%5000==0:
        #     loss_val = 0
        #     for j in range(1018):
        #         x_batch, y_batch = next(val_batch_generator)
        #         val_loss, result1 = sess.run([loss, result], feed_dict={X:x_batch, Y_pred: y_batch, keep_prob: 1.0})
        #         loss_val += val_loss
        #         val_acc = accuracy_score(y_batch, (result1>0.33).astype(int))
        #         acc_val += val_acc
        #     print('step: {}, loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}'\
        #           .format(i, loss_train, loss_val/1018, acc_train/((i+1)*64), acc_val/1018))


def find_best_threshold(y_val, result2):
    print('--------------with threshold--------------------')
    best_thresh, best_f1_score = 0.0, 0.0
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        f1_score1 = f1_score(y_val, (np.array(result2) > thresh).astype(int))
        if f1_score1 > best_f1_score:
            best_f1_score = f1_score1
            best_thresh = thresh
        print("---threshold: %f, all validation f1 score: %f" % (thresh, f1_score1))
    print('------------best result-----------------')
    print('---best thresh: %f, ---best f1 score: %f' % (best_thresh, best_f1_score))
    return best_thresh


def val_predict(sess, X_val):
    result2 = list()
    m = 0
    while m <= 510:
        if m == 510:
            result1 = sess.run(result, feed_dict={X: X_val[m * 128:65307], keep_prob: 1})
        else:
            result1 = sess.run(result, feed_dict={X: X_val[m * 128:128 * (m + 1)], keep_prob: 1})
        result2.extend(result1.tolist())
        m += 1
    return result2


def test_predict(sess, test_data):
    result2 = list()
    m = 0
    while m <= 440:
        if m == 440:
            result1 = sess.run(result, feed_dict={X: test_data[m * 128:56370], keep_prob: 1})
        else:
            result1 = sess.run(result, feed_dict={X: test_data[m * 128:128 * (m + 1)], keep_prob: 1})
        result2.extend(result1.tolist())
        m += 1
    return result2


# predict validation data
result2 = val_predict(sess, X_val)
# find best threshold
best_thresh = find_best_threshold(y_val, result2)
del result2

# predict test data
result2 = test_predict(sess, test_data)
test_label = (np.array(result2) > best_thresh).astype(int)

df = pd.DataFrame()
df["qid"] = df_test["qid"].tolist()
df["prediction"] = test_label
df.to_csv("submission.csv", index=False)

