from preprocessor.document_preprocessor import Preprocessor
from execute.train import Model
from models.encoder import Encoder
from models.decoder import Decoder
from models.tokenizer import Tokenizer  
import tensorflow as tf

def main():
    # Preprocess data
    data = Preprocessor().run()
    print("Data preprocessing completed.")

    # Filter out entries with no questions
    data = data[data['questions'].map(lambda x: len(x) > 0)]
    encoder_texts = data['content'].tolist()
    decoder_texts = [q[0] for q in data['questions'].tolist()]

    # Initialize custom tokenizer
    tokenizer = Tokenizer(encoder_texts + decoder_texts, seq_len=20)

    # Tokenize
    encoder_inputs = tokenizer(encoder_texts)
    decoder_captions = tokenizer(decoder_texts)

    # Split
    split = int(0.8 * len(encoder_inputs))
    train_encoder = encoder_inputs[:split]
    test_encoder = encoder_inputs[split:]
    train_decoder = decoder_captions[:split]
    test_decoder = decoder_captions[split:]

    # Build model
    vocab_size = tokenizer.vectorizer.vocabulary_size()
    encoder = Encoder(vocab_size=vocab_size, hidden_size=128, window_size=20)
    decoder = Decoder(vocab_size=vocab_size, hidden_size=128, window_size=20)
    model = Model(encoder=encoder, decoder=decoder, padding_index=0)
    model.compile(optimizer=tf.keras.optimizers.Adam())

    # Train and evaluate
    model.train(train_decoder, train_encoder, batch_size=32)
    model.test(test_decoder, test_encoder, batch_size=32)

if __name__ == "__main__":
    main()