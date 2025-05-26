import spacy
import tensorflow as tf
from tensorflow import keras
import numpy as np

@keras.saving.register_keras_serializable(package="retriever")
class Retriever:
    def __init__(self, corpus, encoder, tokenizer, num_docs=100, embedding_dim=128): 
        self.corpus = corpus
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.num_docs = num_docs
        self.embedding_dim = embedding_dim
        self.nlp = spacy.load("en_core_web_sm")
        self.corpus_embeddings = self.encode_corpus()

    def encode_corpus(self):
        tokenized = self.tokenizer(self.corpus)
        outputs = self.encoder(tokenized, encoder_output=None)
        embeddings = tf.reduce_mean(outputs, axis=1)
        return embeddings.numpy()

    def encode_query(self, query):
        query_tokens = self.tokenizer([query])
        outputs = self.encoder(query_tokens, encoder_output=None)
        return tf.reduce_mean(outputs, axis=1).numpy()[0]

    def retrieve(self, query):
        entities = self.extract_entity(query)
        query_embedding = self.encode_query(query)

        corpus_norms = np.linalg.norm(self.corpus_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        sims = np.dot(self.corpus_embeddings, query_embedding) / (corpus_norms * query_norm + 1e-8)

        for i, doc in enumerate(self.corpus):
            if any(ent in doc.lower() for ent in entities):
                sims[i] += 0.1  # boost

        top_indices = np.argsort(sims)[-self.num_docs:][::-1]
        return [self.corpus[i] for i in top_indices]

    def extract_entity(self, query):
        doc = self.nlp(query)
        return [ent.text.lower() for ent in doc.ents]

