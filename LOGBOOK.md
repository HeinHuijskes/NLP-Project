### Data preparation

* **Comments in Lyrics**: <br> Noticed that songs contain a lot of comments between brackets, added regex data prep to remove these.

* **Stop word removal**: <br> Removed stopwords using nltk english stopwords.
* **Rare word removal**: <br>
* **Balancing the dataset**: <br> Prepped data to have different levels of balance and categories.
* **Tokenization**: <br> Tokenize using nltk tokenizer. (This was not mentioned in our project proposal?).
* **Normalization**: <br> Normalized by lowercasing all words.
* **Lemmatization or Stemming**: <br> Implemented both lemmatization and stemming with nltk.
* **Vectorization**: <br>


## TODO:
* Check whether any songs are not in English, and should have other stop word removal
* Maybe check to see if there are better ways to stem/lemmatize especially rap lyrics, since it tends to use slang. The paper should also have info on this.
* Maybe check whether we an add a stem step in between to catch words like "goddddd" and reduce it to "god", and see if this actually helps.