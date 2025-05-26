from preprocessor.document_preprocessor import Preprocessor
from execute.train import TrainModel, TrainRetriever
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

    # Tokenize inputs for encoder and decoder
    encoder_inputs = tokenizer(encoder_texts)
    decoder_captions = tokenizer(decoder_texts)

    # Split data
    split = int(0.8 * len(encoder_texts))
    train_encoder_texts = encoder_texts[:split]
    test_encoder_texts = encoder_texts[split:]
    train_decoder_captions = decoder_captions[:split]
    test_decoder_captions = decoder_captions[split:]

    # Build encoder and decoder
    vocab_size = tokenizer.vectorizer.vocabulary_size()
    encoder = Encoder(vocab_size=vocab_size, hidden_size=128, window_size=20)
    decoder = Decoder(vocab_size=vocab_size, hidden_size=128, window_size=20)

    # Train the retriever (encoder) first
    retriever_trainer = TrainRetriever(encoder=encoder, tokenizer=tokenizer)
    epochs = 3
    for epoch in range(epochs):
        # You must prepare positive and negative document pairs for training!
        # Here placeholders: replace with actual batch generation logic
        queries = train_encoder_texts
        pos_docs = train_encoder_texts
        neg_docs = train_encoder_texts[::-1]
        loss = retriever_trainer.train_step(queries, pos_docs, neg_docs)
        print(f"Epoch {epoch + 1}/{epochs} Retriever Loss: {loss.numpy():.4f}")

    # Encode all encoder texts to get retriever embeddings for decoder training
    train_encoder_outputs = encoder(tokenizer(train_encoder_texts))
    test_encoder_outputs = encoder(tokenizer(test_encoder_texts))

    # Initialize TrainModel with only the decoder (encoder outputs are inputs)
    model = TrainModel(decoder=decoder, padding_index=0)
    model.compile(optimizer=tf.keras.optimizers.Adam())

    # Train and test decoder using encoder outputs
    model.train(train_decoder_captions, train_encoder_outputs, batch_size=32)
    model.test(test_decoder_captions, test_encoder_outputs, batch_size=32)

if __name__ == "__main__":
    main()