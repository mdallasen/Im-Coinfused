import spacy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.text import Tokenizer
from models.encoder import Encoder  
import numpy as np

class Retriever: 
    def __init__(self, corpus, encoder, num_docs = 100, embedding_dim = 128): 
        self.corpus = corpus
        self.num_docs = num_docs
        self.embedding_dim = embedding_dim
        self.encoder = encoder
        self.corpus_embeddings = self.encode_corpus()
        self.nlp = spacy.load("en_core_web_sm")

        self.vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=10000,        
            output_mode='int',       
            output_sequence_length=20 
        )

    def tokenize_corpus(self): 
        self.vectorizer.adapt(self.corpus)
        return self.vectorizer(self.constant(self.corpus))

    def encode_corpus(self): 
        tokenized = self.tokenize_corpus()
        outputs = self.encoder(tokenized)
        embeddings = tf.reduce_mean(outputs, axis=1) 
        return embeddings.numpy()

    def extract_entity(self, query): 
        doc = self.nlp(query)
        return [ent.text.lower() for ent in doc.ents]
        
    def encode_query(self, query):
        query_tokens = self.vectorizer(tf.constant([query]))
        outputs = self.encoder(query_tokens)
        return tf.reduce_mean(outputs, axis=1).numpy()[0]

    def retrieve(self, query): 
        entities = self.extract_entity(query)
        query_embedding = self.encode_query(query)

        corpus_norms = np.linalg.norm(self.corpus_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        sims = np.dot(self.corpus_embeddings, query_embedding) / (corpus_norms * query_norm + 1e-8)

        for i, doc in enumerate(self.corpus):
            if any(ent in doc.lower() for ent in entities):
                sims[i] += 0.1

        top_indices = np.argsort(sims)[self.num_docs:][::-1]
        return [self.corpus[i] for i in top_indices]

