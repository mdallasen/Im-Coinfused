import numpy as np
import tensorflow as tf
import keras

class TrainModel(keras.Model):
    def __init__(self, decoder, padding_index=0, **kwargs):
        super().__init__(**kwargs)
        self.decoder = decoder
        self.padding_index = padding_index

    def call(self, inputs, training=False):
        captions, encoder_output = inputs
        return self.decoder(captions, encoder_output, training=training)

    def compile(self, optimizer, loss=None, metrics=None):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss or self.loss_function
        self.metric_fn = metrics or self.accuracy_function

    def train(self, captions, encoder_outputs, batch_size=30):
        print("[INFO] Starting training loop...")

        indices = tf.random.shuffle(tf.range(len(captions)))
        captions = tf.gather(captions, indices)
        encoder_outputs = tf.gather(encoder_outputs, indices)

        num_batches = int(len(captions) / batch_size)
        total_loss = total_seen = total_correct = 0

        for index, end in enumerate(range(batch_size, len(captions) + 1, batch_size)):
            start = end - batch_size
            batch_captions = captions[start:end]
            batch_encoder_outputs = encoder_outputs[start:end]

            decoder_input = batch_captions[:, :-1]
            decoder_labels = batch_captions[:, 1:]

            with tf.GradientTape() as tape:
                probs = self((decoder_input, batch_encoder_outputs), training=True)
                mask = decoder_labels != self.padding_index
                loss = self.loss_fn(probs, decoder_labels, mask)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            accuracy = self.metric_fn(probs, decoder_labels, mask)

            num_preds = tf.reduce_sum(tf.cast(mask, tf.float32))
            total_loss += loss
            total_seen += num_preds
            total_correct += num_preds * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)

            print(f"[Train {index+1:03}/{num_batches}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | Perp: {avg_prp:.2f}")

        print("[INFO] Training complete.\n")
        return avg_prp, avg_acc

    def test(self, captions, encoder_outputs, batch_size=30):
        print("[INFO] Starting evaluation loop...")

        num_batches = int(len(captions) / batch_size)
        total_loss = total_seen = total_correct = 0

        for index, end in enumerate(range(batch_size, len(captions) + 1, batch_size)):
            start = end - batch_size
            batch_captions = captions[start:end]
            batch_encoder_outputs = encoder_outputs[start:end]

            decoder_input = batch_captions[:, :-1]
            decoder_labels = batch_captions[:, 1:]

            probs = self((decoder_input, batch_encoder_outputs), training=False)
            mask = decoder_labels != self.padding_index
            loss = self.loss_fn(probs, decoder_labels, mask)
            accuracy = self.metric_fn(probs, decoder_labels, mask)

            num_preds = tf.reduce_sum(tf.cast(mask, tf.float32))
            total_loss += loss
            total_seen += num_preds
            total_correct += num_preds * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)

            print(f"[Test  {index+1:03}/{num_batches}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | Perp: {avg_prp:.2f}")

        print("[INFO] Evaluation complete.\n")
        return avg_prp, avg_acc

    def loss_function(self, probs, labels, mask):
        masked_labels = tf.boolean_mask(labels, mask)
        masked_probs = tf.boolean_mask(probs, mask)
        scce = tf.keras.losses.sparse_categorical_crossentropy(masked_labels, masked_probs, from_logits=True)
        return tf.reduce_sum(scce)

    def accuracy_function(self, probs, labels, mask):
        preds = tf.argmax(probs, axis=-1, output_type=tf.int32)
        correct = tf.cast(preds == tf.cast(labels, tf.int32), tf.float32)
        masked_correct = tf.boolean_mask(correct, mask)
        return tf.reduce_mean(masked_correct)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "decoder": tf.keras.utils.serialize_keras_object(self.decoder),
            "padding_index": self.padding_index,
        }

    @classmethod
    def from_config(cls, config):
        decoder_config = config.pop("decoder")
        decoder = tf.keras.utils.deserialize_keras_object(decoder_config)
        return cls(decoder, **config)


class TrainRetriever:
    def __init__(self, encoder, tokenizer, margin=0.2):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.margin = margin
        self.optimizer = None
        self.loss_fn = None

    def compile(self, optimizer=None, loss=None):
        self.optimizer = optimizer or tf.keras.optimizers.Adam()
        self.loss_fn = loss or self.contrastive_loss

    def encode(self, texts):
        tokenized = self.tokenizer(texts)
        embeddings = self.encoder(tokenized, encoder_output=None)  # [batch, seq_len, hidden_size]
        pooled = tf.reduce_mean(embeddings, axis=1)  # [batch, hidden_size]
        return pooled

    def contrastive_loss(self, query_embeds, pos_doc_embeds, neg_doc_embeds):
        query_embeds = tf.math.l2_normalize(query_embeds, axis=1)
        pos_doc_embeds = tf.math.l2_normalize(pos_doc_embeds, axis=1)
        neg_doc_embeds = tf.math.l2_normalize(neg_doc_embeds, axis=1)

        pos_sim = tf.reduce_sum(query_embeds * pos_doc_embeds, axis=1)
        neg_sim = tf.reduce_sum(query_embeds * neg_doc_embeds, axis=1)

        loss = tf.maximum(0.0, self.margin - pos_sim + neg_sim)
        return tf.reduce_mean(loss)

    def train(self, queries, pos_docs, neg_docs, batch_size=32):
        print("[INFO] Starting retriever training loop...")

        data_size = len(queries)
        if data_size < batch_size:
            print("[WARNING] Not enough data to form one batch.")
            return float('nan')

        indices = tf.random.shuffle(tf.range(data_size))
        queries = tf.gather(queries, indices)
        pos_docs = tf.gather(pos_docs, indices)
        neg_docs = tf.gather(neg_docs, indices)

        num_batches = data_size // batch_size
        total_loss = 0.0

        for index, end in enumerate(range(batch_size, data_size + 1, batch_size)):
            start = end - batch_size
            batch_queries = queries[start:end]
            batch_pos_docs = pos_docs[start:end]
            batch_neg_docs = neg_docs[start:end]

            with tf.GradientTape() as tape:
                query_embeds = self.encode(batch_queries)
                pos_doc_embeds = self.encode(batch_pos_docs)
                neg_doc_embeds = self.encode(batch_neg_docs)
                loss = self.loss_fn(query_embeds, pos_doc_embeds, neg_doc_embeds)

            grads = tape.gradient(loss, self.encoder.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))
            total_loss += float(loss)

            avg_loss = total_loss / (index + 1)
            print(f"[Retriever Train {index+1:03}/{num_batches}] Loss: {avg_loss:.4f}")

        print("[INFO] Retriever training complete.\n")
        return avg_loss