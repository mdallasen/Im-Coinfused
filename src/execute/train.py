import numpy as np
import tensorflow as tf
import keras

class Model(keras.Model): 

    def __init__(self, encoder, decoder, padding_index=0, **kwargs): 
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.padding_index = padding_index

    def call(self, inputs, training=False):
        captions, encoder_inputs = inputs
        encoded = self.encoder(encoder_inputs, training=training)
        return self.decoder(captions, encoded, training=training)
    
    def compile(self, optimizer, loss=None, metrics=None):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss or self.loss_function
        self.metric_fn = metrics or self.accuracy_function
        
    def train(self, captions, encoder_inputs, batch_size=30):
        indices = tf.random.shuffle(tf.range(len(captions)))
        captions = tf.gather(captions, indices)
        encoder_inputs = tf.gather(encoder_inputs, indices)

        num_batches = int(len(captions) / batch_size)
        total_loss = total_seen = total_correct = 0

        for index, end in enumerate(range(batch_size, len(captions)+1, batch_size)):
            start = end - batch_size
            batch_captions = captions[start:end]
            batch_inputs = encoder_inputs[start:end]

            decoder_input = batch_captions[:, :-1]
            decoder_labels = batch_captions[:, 1:]

            with tf.GradientTape() as tape:
                probs = self((decoder_input, batch_inputs), training=True)
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

            print(f"\r[Train {index+1}/{num_batches}] loss={avg_loss:.3f} acc={avg_acc:.3f} perp={avg_prp:.3f}", end='')

        print()
        return avg_prp, avg_acc

    def test(self, captions, encoder_inputs, batch_size=30):
        num_batches = int(len(captions) / batch_size)
        total_loss = total_seen = total_correct = 0

        for index, end in enumerate(range(batch_size, len(captions)+1, batch_size)):
            start = end - batch_size
            batch_captions = captions[start:end]
            batch_inputs = encoder_inputs[start:end]

            decoder_input = batch_captions[:, :-1]
            decoder_labels = batch_captions[:, 1:]

            probs = self((decoder_input, batch_inputs), training=False)
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

            print(f"\r[Test {index+1}/{num_batches}] loss={avg_loss:.3f} acc={avg_acc:.3f} perp={avg_prp:.3f}", end='')

        print()
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
        config = {
            "encoder": tf.keras.utils.serialize_keras_object(self.encoder),
            "decoder": tf.keras.utils.serialize_keras_object(self.decoder),
            "padding_index": self.padding_index,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        encoder_config = config.pop("encoder")
        decoder_config = config.pop("decoder")
        encoder = tf.keras.utils.deserialize_keras_object(encoder_config)
        decoder = tf.keras.utils.deserialize_keras_object(decoder_config)
        return cls(encoder, decoder, **config)