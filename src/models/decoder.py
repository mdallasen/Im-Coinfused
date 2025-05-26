import tensorflow as tf
from tensorflow import keras
from models.transformer import TransformerBlock, PositionalEncoding

@keras.saving.register_keras_serializable(package="decoder")
class Decoder(keras.Model):
    def __init__(self, vocab_size, hidden_size, window_size, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(window_size, hidden_size)  
        self.transformer_block = TransformerBlock(hidden_size, num_heads=num_heads, multiheaded=True)

        self.classification = tf.keras.layers.Dense(vocab_size)

    def call(self, captions, encoder_output):
        x = self.embedding(captions)
        x = self.pos_encoding(x)  
        out = self.transformer_block(x, encoder_output)
        logits = self.classification(out)  

        return logits