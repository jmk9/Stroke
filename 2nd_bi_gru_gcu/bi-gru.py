import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from importlib import import_module
import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras as KerasAPI
keras: KerasAPI = import_module("tensorflow.keras")
from keras.api._v2 import keras as KerasAPI
from keras import layers
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from string import punctuation
from sklearn.metrics import roc_auc_score, classification_report
from preprocessing import prepro_csv1, prepro_csv2
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

stop_words = set(stopwords.words('english')).union(set(punctuation))
stop_words = stop_words.union(('2.', '1.', '3.', '4.', '5.', '8.', '--'))
tokenizer = TreebankWordTokenizer()

df_, target = prepro_csv1("./data/train_set_no_duplicates.csv")
train_ds = tf.data.Dataset.from_tensor_slices((df_, target))

test_df, t_target = prepro_csv2("./data/test.csv")
test_ds = tf.data.Dataset.from_tensor_slices((test_df, t_target))

def custom_split_fn(string_tensor):
    if tf.is_tensor(string_tensor):
        string_numpy = string_tensor.numpy().decode()
    else:
        string_numpy = string_tensor.decode()
    string_split_lst = [w for w in tokenizer.tokenize(string_numpy) if w not in stop_words]
    return tf.ragged.constant(string_split_lst)

max_length = 600
max_tokens = 25000
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
    standardize="lower_and_strip_punctuation",
    ngrams=2,
    split=custom_split_fn
)

text_only_train_ds = train_ds.map(lambda x, y: x)
text_vectorization.adapt(text_only_train_ds)

test_DS = test_ds.map(
    lambda x, y: (tf.numpy_function(text_vectorization, [x], tf.int64), y),
    num_parallel_calls=4).batch(32)

model = keras.models.load_model("best_weight.h5", compile = False)
predict = model.predict(test_DS, verbose=0)
# predict = prediction.copy()
# prediction[prediction>=0.5] = 1
# prediction[prediction<0.5] = 0

# print(classification_report(t_target, prediction, digits=4))
# print(roc_auc_score(t_target, prediction))

import sys
np.set_printoptions(precision=6, suppress=True, threshold=sys.maxsize)
result = np.concatenate((predict, 1-predict), axis=1)
print(result)