# Deep Learning Homework
For a demo, see `demo.py`.

## Team "deep_learners"
- Kurucz György
- Zeller Márió
- Józsa György

## Topic
Our aim is to train a neural network to be able to generate Hungarian news articles based on the style of multiple news media, namely `origo.hu` and `index.hu`. On supplying the network with an initial prompt (the beginning sentence or paragraph of an article), we expect an output of an entire article.

## Database
We scraped two major hungarian news websites: `index.hu`, and `origo.hu`. What we extracted is the plaintext data of tens of thousands of news articles.
- The dataset for index.hu is available here: [index.zip](https://kuruczgy.com/u/Auwna30x/index.zip) (SHA1: 065f4005cdfd7caad9fde1b08b931d3eedab9755)
- The dataset for origo.hu is available in the repository's `data` folder.

## Method
To get the data from index.hu:
- We used `scrape/index` to generate a list of article URLs.
- Then `scrape/index_cikk` to download articles into separate text files.

To get the data from origo.hu:
- We used this [colab notebook](https://colab.research.google.com/drive/1bo_ql5_60SqLqwxtb88PK0KkemL-qdRO) to end up with one text file per article that contains its plaintext data.

To create final data files for training, validation, and testing:
- We first used `scrape/filter_raw_data` to filter out any strange characters, and aggregate the whole dataset into a single file.
- Then using `scrape/split_data` we split this file into separate training, testing, and validation datasets.

## Documentation
The documentation of the project can be found in the root directory of the project as documentation.pdf
