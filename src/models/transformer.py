import math
import numpy as np
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="transformer_layers")
class AttentionMatrix(keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def call(self, inputs):

        K, Q = inputs
        window_size_queries = Q.get_shape()[1]  
        window_size_keys    = K.get_shape()[1] 
        embedding_size_keys = K.get_shape()[2]

        mask = tf.convert_to_tensor(
            value=np.transpose(np.tril(np.ones((window_size_queries, window_size_keys)) * np.NINF, -1), (1, 0)),
            dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

        if not self.use_mask: 

            top = tf.matmul(Q, K, transpose_b = True)
            bottom = tf.sqrt(tf.cast(K.get_shape()[2], tf.float32))
            att_matrix = tf.nn.softmax(top / bottom, axis = -1)
        
        else: 
            
            top = tf.matmul(Q, K, transpose_b = True)
            bottom = tf.sqrt(tf.cast(K.get_shape()[2], tf.float32))
            inter = (top / bottom ) + atten_mask
            att_matrix = tf.nn.softmax(inter, axis=-1)

        return att_matrix

@keras.saving.register_keras_serializable(package="transformer_layers")
class AttentionHead(keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        self.K = self.add_weight(
            shape = (input_size, output_size), 
            initializer="glorot_uniform",
            trainable = True, 
            name = "K"
        )

        self.Q = self.add_weight(
            shape = (input_size, output_size), 
            initializer="glorot_uniform",
            trainable = True, 
            name = "Q"
        )

        self.V = self.add_weight(
            shape = (input_size, output_size), 
            initializer="glorot_uniform",
            trainable = True, 
            name = "V"
        )
        
        self.att_matrix = AttentionMatrix(use_mask = self.use_mask)

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):

        keys = tf.tensordot(inputs_for_keys, self.K, axes=1)
        values = tf.tensordot(inputs_for_values, self.V, axes=1)
        queries = tf.tensordot(inputs_for_queries, self.Q, axes=1)
        att_matrix = self.att_matrix([keys, queries])
        out = tf.matmul(att_matrix, values)

        return out

@keras.saving.register_keras_serializable(package="transformer_layers")
class MultiHeadedAttention(keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)
        
        self.att_heads = [
            AttentionHead(emb_sz, emb_sz // 3, is_self_attention=use_mask) for _ in range(3)
        ]

        self.dense = keras.layers.Dense(emb_sz)

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):

        attention_outputs = [att_head(inputs_for_keys, inputs_for_values, inputs_for_queries) for att_head in self.att_heads]
        att_head_out = tf.concat(attention_outputs, axis=-1)
        att_head_out = self.dense(att_head_out)

        return att_head_out

@keras.saving.register_keras_serializable(package="transformer_layers")
class TransformerBlock(keras.layers.Layer):
    def __init__(self, emb_sz, multiheaded=True, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        self.feed = keras.Sequential([
            keras.layers.Dense(emb_sz * 4, activation='relu'),
            keras.layers.Dense(emb_sz)
        ])

        self.att = MultiHeadedAttention(emb_sz, use_mask=True)
        self.en_de_att = MultiHeadedAttention(emb_sz, use_mask=False)
        
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, context_sequence):

        att = self.att(inputs, inputs, inputs)
        norm_att = self.layernorm1(inputs + att)  

        en_de_att = self.en_de_att(context_sequence, context_sequence, norm_att)
        norm_en_de_att = self.layernorm2(norm_att + en_de_att) 

        ff = self.feed(norm_en_de_att)
        norm_ff = self.layernorm3(norm_en_de_att + ff)  

        return keras.activations.relu(norm_ff)


@keras.saving.register_keras_serializable(package="transformer_layers", name="positional_encoding")
def positional_encoding(length, depth):

    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(depth)[np.newaxis, :]
    val = pos / 10000 ** (2 * (i // 2) / depth)

    pos_encoding = np.concatenate(
        [np.sin(val[:, 0::2]), np.cos(val[:, 1::2])], axis=-1
    )
    
    return tf.cast(pos_encoding, dtype=tf.float32)


@keras.saving.register_keras_serializable(package="transformer_layers")
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
        self.pos_encoding = positional_encoding(length=window_size, depth=embed_size)[..., :window_size, :]

    def call(self, x):
        
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embed_size, dtype=tf.float32))
        pos_exp = self.pos_encoding[tf.newaxis, :length, :]
        pos_exp = tf.broadcast_to(pos_exp, shape=(x.shape[0], length, self.embed_size))
        x = x + pos_exp

        return x 