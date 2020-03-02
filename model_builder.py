from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy
from keras.models import Model, Input
from keras.layers import Embedding, Dense, Bidirectional, Dropout, LSTM, TimeDistributed
from keras import backend as K
import tensorflow as tf

from Embedding.bert import BertLayer

# Build model
def build_model(max_seq_length, n_tags):
    # Bert Embeddings
    in_id = Input(shape=(max_seq_length,), name="input_ids")
    in_mask = Input(shape=(max_seq_length,), name="input_masks")
    in_segment = Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]
    bert_output = BertLayer(n_fine_tune_layers=10, mask_zero=True, trainable=True)(bert_inputs)

    lstm = Bidirectional(LSTM(units=128, return_sequences=True))(bert_output)
    drop = Dropout(0.4)(lstm)
    dense = TimeDistributed(Dense(128, activation="relu"))(drop)
    crf = CRF(n_tags)
    out = crf(dense)
    model = Model(inputs=bert_inputs, outputs=out)
    model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    model.summary()
    
    
    return model

  
def initialize_vars(sess):
    K.get_session().run(tf.local_variables_initializer())
    K.get_session().run(tf.global_variables_initializer())
    K.get_session().run(tf.tables_initializer())