# Deep Learning házi

## Csapat
- Kurucz György
- Zeller Márió
- Józsa György

# Téma
TODO

# Database
We scraped two major hungarian news websites: `index.hu`, and `origo.hu`. The two datasets are available here:
- [index.zip](https://kuruczgy.com/u/Auwna30x/index.zip)

## Method
- We used `scrape/index` to generate a list of article URLs.
- `scrape/index_cikk` is a script to download articles into separate text files.
- `scrape/filter_raw_data` filters out any strange characters, and aggregates the whole dataset into a single file.
- `scrape/split_data` splits this file into separate training, testing, and validation datasets.
