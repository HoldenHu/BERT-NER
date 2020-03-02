from keras.layers import Layer
import tensorflow_hub as hub
from keras import backend as K

bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

class BertLayer(Layer):
    def __init__(self, n_fine_tune_layers=10, mask_zero=False, trainable=True, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = trainable
        self.output_size = 768
        self.mask_zero=mask_zero
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            bert_path,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        
        # TRAINABLE PARAMS: TODO: Test that if have time
#         trainable_vars = self.bert.variables
        
        # Remove unused layers
#         trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
        
        # Select how many layers to fine tune
#         trainable_vars = trainable_vars[-self.n_fine_tune_layers :]
#         
        # Add to trainable weights
#         for var in trainable_vars:
#             print(var)
#             self._trainable_weights.append(var)
        
         # Remove unused layers and set trainable parameters
        self.trainable_weights += [var for var in self.bert.variables
                                   if not "/cls/" in var.name and not "/pooler/" in var.name][-self.n_fine_tune_layers :]

        # Add non-trainable weights
        for var in self.bert.variables:
            if var not in self.trainable_weights:
                self.non_trainable_weights.append(var)
                
        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask,
                           segment_ids=segment_ids)
        result = self.bert(inputs=bert_inputs, signature="tokens",
                           as_dict=True)["sequence_output"]
        result = K.reshape(result, (-1,inputs[0].shape[1],768))
        return result

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], self.output_size)
      
    def compute_mask(self, inputs, mask=None):
      input_ids, input_mask, segment_ids = inputs
      if not self.mask_zero:
          return None
      return K.not_equal(input_ids, 0)