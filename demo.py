#!/usr/bin/env python3
import hungarian_news
raw_training_data = hungarian_news.load_raw_training_data()
tokenizer = hungarian_news.get_tokenizer(raw_training_data)
training_data = hungarian_news.build_training_data(raw_training_data, tokenizer)

build_model = hungarian_news.get_model_builder(tokenizer)
model = build_model(None)
hungarian_news.train(model, training_data, 1)

hungarian_news.generate(model, tokenizer, "Magyarország ", max_len = 32, top_k = 3)
