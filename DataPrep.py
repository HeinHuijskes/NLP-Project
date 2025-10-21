import pandas as pd
from regex import regex as re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import FreqDist
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')


def remove_notes(docs: list[str]) -> list[str]:
    result = []
    # RegEx pattern to target all notes in brackets, like [Chorus:] or [Kanye West Sings]
    pattern = r'\[[^\n\[\]]+\]'
    for doc in docs:
        # Remove all bracket notes
        new_doc = re.sub(pattern, '', doc)
        result.append(new_doc)
    return result


def remove_punctuation(docs: list[str]) -> list[str]:
    '''Remove useless punctuation from all documents'''
    result = []
    # RegEx pattern to target all characters that are NOT:
    # a letter, underscore, dash, space, or newline
    pattern = r"[^\w' \n\-\_]"
    for doc in docs:
        # Reduce repeated newlines to singular newlines
        new_doc = re.sub(r'\n+', ' ', doc)
        # Remove punctuation 
        new_doc = re.sub(pattern, ' ', new_doc)
        result.append(new_doc)
    return result


def normalize(docs: list[str]) -> list[str]:
    '''Simply lowercase all words'''
    return [doc.lower() for doc in docs]


def tokenize(docs: list[str]) -> list[list[str]]:
    '''Tokenize documents using nltk'''
    result = []
    for doc in docs:
        new_doc = word_tokenize(doc)
        result.append(new_doc)
    return result


def remove_stopwords(docs: list[list[str]]) -> list[list[str]]:
    '''Remove stopwords in documents using nltk english stopwords'''
    stop_words = stopwords.words('english')
    result = []
    for doc in docs:
        new_doc = [word for word in doc if word not in stop_words]
        result.append(new_doc)
    return result


def lemmatize(docs: list[list[str]]) -> list[list[str]]:
    '''Lemmatize documents usting nltk WordNetLemmatizer'''
    lemmatizer = WordNetLemmatizer()
    result = []
    for doc in docs:
        new_doc = [lemmatizer.lemmatize(word) for word in doc]
        result.append(new_doc)
    return result


def stem(docs: list[list[str]]) -> list[list[str]]:
    '''Stem documents usting nltk PorterStemmer'''
    stemmer = PorterStemmer()
    result = []
    for doc in docs:
        new_doc = [stemmer.stem(word) for word in doc]
        result.append(new_doc)
    return result


def vocabulary(docs: list[list[str]]) -> tuple[list[list[str]], list[str]]:
    '''Return unique words per document, and a total vocabulary'''
    vocab = set()
    result = []
    for doc in docs:
        for word in doc:
            vocab.add(word)
        result.append(list(set(doc)))
    return result, list(vocab)


def write_out(data, columns, location):
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f'output/{location}')


def remove_rare_words(docs: list[list[str]], limit=1, other_limit=0, location='', debug=True, return_count=False) -> list[list[str]]:
    '''Remove words that occur in less than a given proportion of documents'''
    unique_docs, vocab = vocabulary(docs)
    all_words = []
    for doc in unique_docs:
        all_words += doc
    size = len(all_words)
    distribution = FreqDist(all_words)
    distribution_words = list(distribution)

    if debug:
        print(f'Total words: {size}')
        print(f'Unique words: {len(vocab)}')
        print('Most frequent:')
        print('; '.join([f'{word}: {int(distribution.freq(word) * size)}' for word in distribution_words[:10]]))
        print('Least frequent:')
        print(' - '.join([f'{word}: {int(distribution.freq(word) * size)}' for word in distribution_words[-10:]]))

    result = []
    removed = []
    for doc in docs:
        new_doc = [word for word in doc if not other_limit < int(distribution.freq(word) * size) < limit]
        removed += [word for word in doc if other_limit < int(distribution.freq(word) * size) < limit]
        result.append(new_doc)
    new_vocab = [word for word in vocab if not other_limit < int(distribution.freq(word) * size) < limit]

    output = []
    for word in vocab:
        output.append([word, int(distribution.freq(word) * size), int(distribution.freq(word) * size) > limit])
    if location != '':
        write_out(output, ['word', 'occurence', 'included'], location)

    if debug:
        print(f'\nRemoved all words occuring {limit} or less times')
        print(f'Reduced vocab from {len(vocab)} to {len(new_vocab)} words')

    if not return_count:
        return result
    else:
        return [len(vocab), len(new_vocab), result]


def preprocess(docs: list[str], stem_words=True, limit=0, other_limit=0, debug=True, return_count=False) -> list[str]:
    '''Apply all preprocessing steps to the given docs'''
    if debug: print("Preprocessing data")
    if debug: print("Removing notes in [brackets]")
    noteless = remove_notes(docs)
    if debug: print("Removing punctuation")
    just_words = remove_punctuation(noteless)
    if debug: print("Normalizing")
    normalized = normalize(just_words)
    if debug: print("Tokenizing")
    tokens = tokenize(normalized)
    if debug: print("Removing stopwords")
    no_stopwords = remove_stopwords(tokens)

    if stem_words:
        if debug: print("Stemming")
        documents = stem(no_stopwords)
    else:
        if debug: print("Lemmatizing")
        documents = lemmatize(no_stopwords)

    if limit > 0 or return_count:
        if debug: print("Removing rare words")
        documents = remove_rare_words(documents, limit=limit, other_limit=other_limit, debug=debug, return_count=return_count)

    if debug: print("Finished data preparation!")
    if not return_count:
        return [' '.join(doc) for doc in documents]
    else:
        old, total, documents = documents
        return old, total, [' '.join(doc) for doc in documents]


if __name__ == "__main__":
    genres = ['rap', 'rock', 'pop']
    limit = '1000'
    filename = f'data/song_lyrics_reduced_{"_".join(genres)}_{limit}.csv'

    print("Loading csv")
    df = pd.read_csv(filename)
    documents = df['lyrics']
    print("Loaded succesfully")
    preprocess(documents, debug=True)