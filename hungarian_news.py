# some code based on: https://www.tensorflow.org/tutorials/text/transformer
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_probability as tfp

import time
import numpy as np
import matplotlib.pyplot as plt

import transformer as trf

import kerastuner

import os
import json

# training parameters
BATCH_SIZE = 16
MAX_LENGTH = 256

rel = 0

def load_raw_training_data(key = 'origo'):
    print("Loading raw training data into memory")
    def load_data(filename):
        with open(filename, 'r') as f:
            return tf.data.Dataset.from_tensor_slices(list([i.strip().encode() for i in f]))
    train_examples, val_examples = load_data(f'data/{key}_train.txt'), load_data(f'data/{key}_valid.txt')
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

    ds_val = raw_val.map(tf_encode)
    ds_val = ds_val.map(cut_max_length)
    ds_val = ds_val.padded_batch(BATCH_SIZE)
    return (ds_train, ds_val)

def get_range_hp():
    hp = kerastuner.engine.hyperparameters.HyperParameters()
    hp.Int('num_layers', min_value=1, max_value=4),
    hp.Int('d_model_per_heads', min_value=32, max_value=128, step=64),
    hp.Int('num_heads', min_value=2, max_value=6),
    hp.Int('dff', min_value=32, max_value=512, step=32),
    return hp
def get_fixed_hp():
    # Values obtained by hyperparameter optimization
    hp = kerastuner.engine.hyperparameters.HyperParameters()
    hp.Fixed('num_layers', 2),
    hp.Fixed('d_model_per_heads', 128),
    hp.Fixed('num_heads', 6),
    hp.Fixed('dff', 256),
    return hp

def get_model_builder(tokenizer):
    def build_model(hp):
        # transformer hyperparameters
        # num_layers: number of decoder layers
        # d_model: embedding dimension
        # num_heads: number of attention heads
        # dff: neurons in feed-forward sublayers
        dropout_rate = 0.1 # TODO: hp

        target_vocab_size = tokenizer.vocab_size + 2
        model = trf.Transformer(
            hp.get('num_layers'),
            hp.get('d_model_per_heads') * hp.get('num_heads'),
            hp.get('num_heads'),
            hp.get('dff'),
            target_vocab_size,
            pe_target=target_vocab_size,
            rate=dropout_rate
        )
        model.compile()
        return model
    return build_model

def get_tuner(build_model):
    tuner = kerastuner.tuners.Hyperband(
        build_model,
        hyperparameters = get_range_hp(),
        objective = 'val_accuracy',
        max_epochs = 10,
        directory = 'hyperopt_output',
        project_name = 'hungarian_news_hyperband'
    )
    return tuner

def hp_optim(tuner, training_data):
    train_dataset, val_dataset = training_data
    tuner.search(
        train_dataset,
        epochs = 10,
        validation_data = val_dataset,
        steps_per_epoch = 100,
        validation_steps = 20,
    )

class EpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, fn):
        self.fn = fn
    def on_epoch_begin(self, epoch, logs=None):
        self.fn(epoch, logs)
    def on_epoch_end(self, epoch, logs=None):
        with open('history_log.json', 'a') as f:
            f.write(f'epoch {epoch}: ')
            f.write(json.dumps(logs))
            f.write('\n')

def train(model, tokenizer, training_data, epochs = 10, load = True, steps_per_epoch = None):
    print(f'training; devices: {tf.config.list_physical_devices()}')
    train_dataset, val_dataset = training_data

    checkpoint_path = './checkpoints/model_best'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor = 'val_accuracy',
        save_best_only = True,
        mode = 'max'
    )

    def on_epoch(epoch, logs):
        prompt = ""
        max_len = 32
        top_k = 3
        result, attention_weights = evaluate(model, tokenizer, prompt, max_len, top_k)
        predicted_sentence = tokenizer.decode([i for i in result if i < tokenizer.vocab_size])
        print(f'epoch {epoch} test predict: {predicted_sentence}')
        with open('test_predict.txt', 'a') as f:
            f.write(f'epoch {epoch}: {predicted_sentence}\n')

    if load:
        model.load_weights(checkpoint_path)
    history = model.fit(
        train_dataset,
        validation_data = val_dataset,
        epochs = epochs,
        steps_per_epoch = steps_per_epoch,
        validation_steps = 100 if steps_per_epoch is None else 1,
        callbacks = [ checkpoint, EpochCallback(on_epoch) ]
    )
    with open('trainHistory.json', 'w') as f:
        f.write(json.dumps(history.history))
    return history

def evaluate(transformer, tokenizer, prompt, max_len, top_k):
    decoder_input = [tokenizer.vocab_size] + tokenizer.encode(prompt)
    output = tf.expand_dims(decoder_input, 0)

    for i in range(max_len):
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(output, training = False)

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

def generate_test_prompts(model, tokenizer, max_len = 32, top_k = 3):
    with open('data/test_prompts.txt', 'r') as f:
        for line in f:
            if line[0] == '#': continue
            line = line.strip()
            if len(line) == 0: continue
            generate(model, tokenizer, line + ' ', max_len = max_len, top_k = top_k)
