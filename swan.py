import tensorflow as tf
import sys
from tensorflow import keras


def symmetric_mask_self(shape, context):
    """Creates a symmetric boolean mask over the last 2 dimensions."""
    row_index = tf.math.cumsum(
        tf.ones(shape=shape, dtype=tf.int32), axis=-2)
    col_index = tf.math.cumsum(
        tf.ones(shape=shape, dtype=tf.int32), axis=-1)
    return tf.abs(row_index - col_index) <= context

def causal_mask(shape, context):
    """Creates a symmetric boolean mask over the last 2 dimensions."""
    row_index = tf.math.cumsum(
        tf.ones(shape=shape, dtype=tf.int32), axis=-2)
    col_index = tf.math.cumsum(
        tf.ones(shape=shape, dtype=tf.int32), axis=-1)
    #return (row_index - col_index) <= context
    return tf.math.logical_and((row_index - col_index) <= context, (row_index - col_index)>=0)
    
@tf.keras.utils.register_keras_serializable()
class MHAttn(tf.keras.layers.Layer):
    def __init__(self, n_heads, head_size, context):
        super(MHAttn, self).__init__()
        self.n_heads = n_heads
        self.head_size = head_size
        self.context = context

    def build(self, input_shape):
        initializer = tf.keras.initializers.GlorotUniform()
        regularizer = tf.keras.regularizers.L2(1e-5)
        
        # self.WQ = self.add_weight("WQ", shape=(self.n_heads, input_shape[-1], self.head_size), 
        #         initializer=initializer, regularizer=regularizer, trainable=True)
        # Correctly passing arguments to add_weight with named parameters
        self.WQ = self.add_weight(
            name="WQ", 
            shape=(self.n_heads, input_shape[-1], self.head_size), 
            initializer=initializer, 
            regularizer=regularizer, 
            trainable=True
        )
        # self.WV = self.add_weight("WV", shape=(self.n_heads, input_shape[-1], self.head_size), 
        #         initializer=initializer, regularizer=regularizer, trainable=True)
        # self.WU = self.add_weight("WU", shape=(self.n_heads*self.head_size, input_shape[-1]), 
        #         initializer=initializer, regularizer=regularizer, trainable=True)

        self.WV = self.add_weight(
            name="WV", 
            shape=(self.n_heads, input_shape[-1], self.head_size), 
            initializer=initializer, 
            regularizer=regularizer, 
            trainable=True
        )
        
        self.WU = self.add_weight(
            name="WU", 
            shape=(self.n_heads * self.head_size, input_shape[-1]), 
            initializer=initializer, 
            regularizer=regularizer, 
            trainable=True
        )
       
    def call(self, inputs):
        input_shape=tf.shape(inputs)
        bool_mask=symmetric_mask_self((input_shape[1], input_shape[1]), context=self.context)
        context_mask = 1.e9*tf.cast(tf.math.logical_not(bool_mask), dtype=tf.float32)

        inputs = tf.expand_dims(inputs, axis=1)
        Q = tf.matmul(inputs, self.WQ)
        # print(Q.shape,"Q shape",inputs.shape)
        # sys.exit()
        V = tf.matmul(inputs, self.WV)
        S = tf.matmul(Q, V, transpose_b=True)
        S = S/tf.math.sqrt(tf.cast(self.head_size, tf.float32))  
        W = tf.keras.activations.softmax(S-context_mask, axis=-1)
        C = tf.matmul(W, V)
       
        C = tf.concat([C[:,head,:,:] for head in range(self.n_heads)], axis=-1)

   
        A = tf.matmul(C, self.WU)
        return A
    
    def get_config(self):
        return {"n_heads": self.n_heads, "head_size": self.head_size, "context": self.context}
