## Download cornell dataset

```bash
mkdir -p cornell
curl -k -L -o ./movie-dialog-corpus.zip   https://www.kaggle.com/api/v1/datasets/download/Cornell-University/movie-dialog-corpus
unzip -d cornell movie-dialog-corpus.zip
```

This will create in the *cornell* subdirectories these files:

* README.txt 
* movie_characters_metadata.tsv
* movie_conversations.tsv (used by this example script)
* movie_lines.tsv (used by this example)
* movie_titles_metadata.tsv
* raw_script_urls.tsv

## Run the scripts


```bash
# Option 1: Train on Cornell Movie Dialog Corpus
python train_model.py --cornell /path/to/cornell/corpus --output movie_model.pkl

# Option 2: Train on your own documents
python train_model.py --documents /path/to/document/directory --output custom_model.pkl
```


