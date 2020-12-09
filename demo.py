#!/usr/bin/env python3
import hungarian_news
raw_training_data = hungarian_news.load_raw_training_data()
tokenizer = hungarian_news.get_tokenizer(raw_training_data)
training_data = hungarian_news.build_training_data(raw_training_data, tokenizer)

build_model = hungarian_news.get_model_builder(tokenizer)
# hungarian_news.hp_optim(build_model, training_data)

hp = hungarian_news.get_fixed_hp()
model = build_model(hp)
hungarian_news.train(model, training_data, 1)

# hungarian_news.generate(model, tokenizer, "Magyarország ", max_len = 32, top_k = 3)
