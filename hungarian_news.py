# some code based on: https://www.tensorflow.org/tutorials/text/transformer
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_probability as tfp

import time
import numpy as np
import matplotlib.pyplot as plt

import transformer as trf

import os

# training parameters
BATCH_SIZE = 32
MAX_LENGTH = 256

rel = 0

def load_raw_training_data():
    print("Loading raw training data into memory")
    def load_data(filename):
        with open(filename, 'r') as f:
            return tf.data.Dataset.from_tensor_slices(list([i.strip().encode() for i in f]))
    train_examples, val_examples = load_data('data/origo_train.txt'), load_data('data/origo_valid.txt')
    return (train_examples, val_examples)

def get_tokenizer(raw_training_data):
    tokenizer_file = "tokenizer_hu"
    if os.path.isfile(tokenizer_file + '.subwords'):
        print("Loading tokenizer from file")
        tokenizer_hu = tfds.deprecated.text.SubwordTextEncoder.load_from_file(tokenizer_file)
    else:
        print("Building tokenizer...")
        corpus = list((i.numpy() for i in raw_training_data[0]))[:1000]
        tokenizer_hu = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(corpus, target_vocab_size=2**13)
        tokenizer_hu.save_to_file(tokenizer_file)
        print("Tokenizer built!")
    return tokenizer_hu

def build_training_data(raw_training_data, tokenizer_hu):
    raw_train, raw_val = raw_training_data

    def encode(lang1):
        lang1 = [tokenizer_hu.vocab_size] + tokenizer_hu.encode(lang1.numpy()) + [tokenizer_hu.vocab_size+1]
        return lang1, [1, 2, 3]
    def tf_encode(hu):
      result_hu, _ = tf.py_function(encode, [hu], [tf.int64, tf.int64])
      result_hu.set_shape([None])

      return result_hu
    def cut_max_length(x, max_length=MAX_LENGTH):
      x = tf.pad(x, [[0, max_length]])
      x = tf.slice(x, [0], [max_length - 2])
      x = tf.pad(x, [[0, 2]])
      return x

    ds_train = raw_train.map(tf_encode)
    ds_train = ds_train.map(cut_max_length)
    ds_train = ds_train.cache()
    BUFFER_SIZE = 5000
    ds_train = ds_train.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_val = raw_val.map(tf_encode)
    ds_val = ds_val.map(cut_max_length)
    ds_val = ds_val.padded_batch(BATCH_SIZE)
    return (ds_train, ds_val)

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = trf.create_padding_mask(inp)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = trf.create_padding_mask(tar)

  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by
  # the decoder.
  look_ahead_mask = trf.create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = trf.create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask

def get_model_builder(tokenizer):
    def build_model(hp):
        # transformer hyperparameters
        num_layers = 4 # number of decoder layers
        d_model = 256 # embedding dimension
        num_heads = 4 # number of attention heads
        dff = 256 # neurons in feed-forward sublayers
        dropout_rate = 0.1

        target_vocab_size = tokenizer.vocab_size + 2
        return trf.Transformer(
            num_layers, d_model, num_heads, dff,
            target_vocab_size,
            pe_target=target_vocab_size,
            rate=dropout_rate
        )
    return build_model

def train(model, training_data, epochs = 10):
    print(f'training; devices: {tf.config.list_physical_devices()}')
    train_dataset, val_dataset = training_data

    # define loss and accuracy functions
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    def accuracy_function(real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)
        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

    # TODO: get from HP
    num_layers = 6 # number of decoder layers
    d_model = 512 # embedding dimension
    num_heads = 8 # number of attention heads
    dff = 2048 # neurons in feed-forward sublayers

    learning_rate = trf.CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    model_key = f'hungarian_news-{num_layers}-{d_model}-{num_heads}-{dff}-{rel}'
    checkpoint_path = f'./checkpoints/{model_key}'
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        inp = tf.zeros(tf.shape(tar_inp), tf.int64)

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = model(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    for epoch in range(epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, inp) in enumerate(train_dataset):
            train_step(inp)

            if batch % 50 == 0:
                print(
                    f'Epoch {epoch+1} Batch {batch} ' +
                    f'Loss {train_loss.result():.4f} ' +
                    f'Accuracy {train_accuracy.result():.4f}')

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

        print(
            f'Epoch {epoch+1} ' +
            f'Loss {train_loss.result():.4f} ' +
            f'Accuracy {train_accuracy.result():.4f}')
        print(f'Time taken for 1 epoch: {(time.time() - start):.2f} secs\n')
    return model

def evaluate(transformer, tokenizer, prompt, max_len, top_k):
    # TODO: remove input
    inp_sentence = tf.zeros([20], tf.int64)
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [tokenizer.vocab_size] + tokenizer.encode(prompt)
    output = tf.expand_dims(decoder_input, 0)

    for i in range(max_len):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        values, indices = tf.math.top_k(predictions, k=top_k)
        indices = indices.numpy().reshape((-1,))
        values = values.numpy().reshape((-1,))
        dist = tfp.distributions.Categorical(probs=values)
        index = dist.sample().numpy()
        predicted_id = [[ indices[index] ]]
        # predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer.vocab_size+1:
          return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

def generate(transformer, tokenizer, prompt, max_len = 32, top_k = 3):
    result, attention_weights = evaluate(transformer, tokenizer, prompt, max_len, top_k)

    # print(f'Tokens: {result}')
    predicted_sentence = tokenizer.decode([i for i in result if i < tokenizer.vocab_size])
    print(f'Prompt: {prompt}')
    print(f'Generated: {predicted_sentence}')
