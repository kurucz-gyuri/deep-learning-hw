#!/usr/bin/env python3

# import module
import hungarian_news

# load training data & tokenizer
raw_training_data = hungarian_news.load_raw_training_data(key = 'index')
tokenizer = hungarian_news.get_tokenizer(raw_training_data)
training_data = hungarian_news.build_training_data(raw_training_data, tokenizer)

# build model
build_model = hungarian_news.get_model_builder(tokenizer)
hp = hungarian_news.get_fixed_hp()
model = build_model(hp)

# train model
# hungarian_news.train(model, tokenizer, training_data, 10, steps_per_epoch = None)

# generate an example for an empty prompt
hungarian_news.generate(model, tokenizer, "", max_len = 32, top_k = 3)
