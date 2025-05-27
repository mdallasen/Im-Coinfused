from data.wiki_scrapper import WikiScapper
from preprocessor.document_preprocessor import Preprocessor
from execute.train import TrainModel, TrainRetriever
from models.encoder import Encoder
from models.decoder import Decoder
from models.tokenizer import Tokenizer  
from utils.utils import sample_negatives
import tensorflow as tf

def main():
    # Load and preprocess dataset
    print("[INFO] Starting data preprocessing...")
    WikiScapper.extract()
    data = Preprocessor().run()
    print("[INFO] Data preprocessing completed.")

    encoder_texts = data['questions'].tolist()
    decoder_texts = data['content'].tolist()

    print(encoder_texts[:5])  
    print(decoder_texts[:5]) 

    # Filter empty strings in combined corpus
    corpus = encoder_texts + decoder_texts
    corpus = [text for text in corpus if isinstance(text, str) and text.strip()]
    print(corpus[:5]) 
    if not corpus:
        raise ValueError("Corpus is empty after filtering, cannot adapt tokenizer.")

    # Initialize tokenizer on both content and questions
    tokenizer = Tokenizer(corpus, seq_len=5)

    # Split into training and test sets
    split_idx = int(0.8 * len(encoder_texts))
    if split_idx == 0 or split_idx == len(encoder_texts):
        raise ValueError("Train/test split index invalid. Check dataset size.")

    train_enc_texts = encoder_texts[:split_idx]
    test_enc_texts = encoder_texts[split_idx:]
    train_dec_texts = decoder_texts[:split_idx]
    test_dec_texts = decoder_texts[split_idx:]

    print(train_enc_texts)
    print(test_enc_texts)
    print(train_dec_texts)
    print(test_dec_texts)

    # Build vocab and models
    vocab_size = tokenizer.vectorizer.vocabulary_size()
    encoder = Encoder(vocab_size=vocab_size, hidden_size=128, window_size=20)
    decoder = Decoder(vocab_size=vocab_size, hidden_size=128, window_size=20)

    # Initialize and compile retriever
    retriever = TrainRetriever(encoder=encoder, tokenizer=tokenizer, margin=0.2)
    retriever.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

    # Prepare negative samples by reversing train_enc_texts (simple neg sampling)
    neg_enc_texts = sample_negatives(train_enc_texts, encoder_texts, num_negatives=1)
    print(neg_enc_texts)

    print("[INFO] Starting retriever training...")
    retriever.train(
        queries=tf.constant(tokenizer(train_enc_texts)),
        pos_docs=tf.constant(tokenizer(train_dec_texts)),
        neg_docs=tf.constant(tokenizer(neg_enc_texts))
    )

    print("[INFO] Encoding training and test texts...")
    train_encoder_embeddings = encoder(tokenizer(train_enc_texts), encoder_output=None)
    test_encoder_embeddings = encoder(tokenizer(test_enc_texts), encoder_output=None)

    print("[INFO] Starting decoder training...")
    train_captions = tokenizer(train_dec_texts)
    test_captions = tokenizer(test_dec_texts)

    # Convert captions to tf.constant for model input
    train_captions = tf.constant(train_captions)
    test_captions = tf.constant(test_captions)

    model = TrainModel(decoder=decoder, padding_index=0)
    model.compile(optimizer=tf.keras.optimizers.Adam())

    model.train(train_captions, train_encoder_embeddings, batch_size=32)
    model.test(test_captions, test_encoder_embeddings, batch_size=32)

    print("[INFO] Training pipeline complete.")

if __name__ == "__main__":
    main()