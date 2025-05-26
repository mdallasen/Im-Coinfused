from data.wiki_scrapper import WikiScapper
from preprocessor.document_preprocessor import Preprocessor
from execute.train import TrainModel, TrainRetriever
from models.encoder import Encoder
from models.decoder import Decoder
from models.tokenizer import Tokenizer  
import tensorflow as tf

def main():
    # Load and preprocess dataset
    print("[INFO] Starting data preprocessing...")
    WikiScapper.extract()
    data = Preprocessor().run()
    print("[INFO] Data preprocessing completed.")

    # Filter entries that have questions
    data = data[data['questions'].map(lambda qlist: len(qlist) > 0)]
    encoder_texts = data['content'].tolist()
    decoder_texts = [q[0] for q in data['questions'].tolist()]  # First question per content

    # Initialize tokenizer on both content and questions
    tokenizer = Tokenizer(encoder_texts + decoder_texts, seq_len=20)

    # Split into training and test sets
    split_idx = int(0.8 * len(encoder_texts))
    train_enc_texts = encoder_texts[:split_idx]
    test_enc_texts = encoder_texts[split_idx:]
    train_dec_texts = decoder_texts[:split_idx]
    test_dec_texts = decoder_texts[split_idx:]

    # Build vocab and models
    vocab_size = tokenizer.vectorizer.vocabulary_size()
    encoder = Encoder(vocab_size=vocab_size, hidden_size=128, window_size=20)
    decoder = Decoder(vocab_size=vocab_size, hidden_size=128, window_size=20)

    retriever = TrainRetriever(encoder=encoder, tokenizer=tokenizer, margin=0.2)
    retriever.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

    neg_enc_texts = train_enc_texts[::-1]

    print("[INFO] Starting retriever training...")
    retriever.train(
        queries=tf.constant(train_enc_texts),
        pos_docs=tf.constant(train_enc_texts),
        neg_docs=tf.constant(neg_enc_texts),
        batch_size=32
    )

    print("[INFO] Encoding training and test texts...")
    train_encoder_embeddings = encoder(tokenizer(train_enc_texts), encoder_output=None)
    test_encoder_embeddings = encoder(tokenizer(test_enc_texts), encoder_output=None)

    print("[INFO] Starting decoder training...")
    train_captions = tokenizer(train_dec_texts)
    test_captions = tokenizer(test_dec_texts)

    model = TrainModel(decoder=decoder, padding_index=0)
    model.compile(optimizer=tf.keras.optimizers.Adam())

    model.train(train_captions, train_encoder_embeddings, batch_size=32)
    model.test(test_captions, test_encoder_embeddings, batch_size=32)

    print("[INFO] Training pipeline complete.")

if __name__ == "__main__":
    main()
