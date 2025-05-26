import tensorflow as tf

class Tokenizer:
    def __init__(self, corpus, seq_len=20, max_tokens=10000):
        self.vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode='int',
            output_sequence_length=seq_len
        )
        self.vectorizer.adapt(corpus)

    def __call__(self, texts):
        return self.vectorizer(tf.constant(texts))

    def decode(self, token_ids):
        inverse_vocab = dict(enumerate(self.vectorizer.get_vocabulary()))
        return [" ".join([inverse_vocab.get(i, "") for i in row if i != 0]) for row in token_ids.numpy()]