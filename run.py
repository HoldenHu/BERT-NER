import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from bert.tokenization import FullTokenizer
from tqdm import tqdm_notebook
from keras import backend as K

from preprocessing import *
from bert import *
from model_builder import *

### DataLoader ###
# import deeppavlov
# from deeppavlov.core.data.utils import download_decompress
# download_decompress('http://files.deeppavlov.ai/deeppavlov_data/conll2003_v2.tar.gz', 'data/')

from deeppavlov.dataset_readers.conll2003_reader import Conll2003DatasetReader
dataset = Conll2003DatasetReader().read('data/')
print("train",len(dataset['train']))
print("test",len(dataset['test']))
print("valid",len(dataset['valid']))


train_words, train_tags = [], []
for tpl in dataset['train']:
    train_words.append(tpl[0])
    train_tags.append(tpl[1])

tags = set([])

for ts in train_tags:
  for i in ts:
    tags.add(i)
tags = list(tags)
tag2idx = {t: i+1 for i, t in enumerate(list(tags))}
print("tag2idx",tag2idx)
tag2idx["-PAD-"] = 0 # for the mask zero
n_tags = len(tag2idx)
###################

# Initialize session
sess = tf.Session()

# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
max_seq_length = 256

train_tag_ids = [list(map(lambda x: tag2idx[x], sample)) for sample in train_tags]

# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module(bert_path,sess)

# Convert data to InputExample format
train_examples = convert_text_to_examples(train_words, train_tag_ids)

# Convert to features
(train_input_ids, train_input_masks, train_segment_ids, train_tag_ids 
) = convert_examples_to_features(tokenizer, train_examples, tag2idx, n_tags, max_seq_length=max_seq_length)

model = build_model(max_seq_length, n_tags)

# Instantiate variables
initialize_vars(sess)

history = model.fit(
    [train_input_ids, train_input_masks, train_segment_ids], 
    train_tag_ids,
    validation_split=0.2,
    epochs=20,
    batch_size=128,
)