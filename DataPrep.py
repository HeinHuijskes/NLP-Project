import pandas as pd
from regex import regex as re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import FreqDist
from nltk.corpus import stopwords
nltk.download('stopwords')


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
    pattern = r"[^\w' \n]"
    for doc in docs:
        # Reduce repeated newlines to singular newlines
        new_doc = re.sub(r'\n+', '\n', doc)
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


def remove_rare_words(docs: list[list[str]], limit=1) -> list[list[str]]:
    '''Remove words that occur in less than a given proportion of documents'''
    unique_docs, vocab = vocabulary(docs)
    all_words = []
    for doc in unique_docs:
        all_words += doc
    size = len(all_words)
    distribution = FreqDist(all_words)
    distribution_words = list(distribution)

    print(f'Total words: {size}')
    print(f'Unique words: {len(vocab)}')
    print('Most frequent:')
    print('; '.join([f'{word}: {int(distribution.freq(word) * size)}' for word in distribution_words[:10]]))
    print('Least frequent:')
    print(' - '.join([f'{word}: {int(distribution.freq(word) * size)}' for word in distribution_words[-10:]]))

    result = []
    removed = []
    for doc in docs:
        new_doc = [word for word in doc if int(distribution.freq(word) * size) > limit]
        removed += [word for word in doc if int(distribution.freq(word) * size) <= limit]
        result.append(new_doc)
    new_vocab = [word for word in vocab if int(distribution.freq(word) * size) > limit]

    output = []
    for word in vocab:
        output.append([word, int(distribution.freq(word) * size), int(distribution.freq(word) * size) > limit])
    write_out(output, ['word', 'occurence', 'included'])

    print(f'\nRemoved all words occuring {limit} or less times')
    print(f'Reduced vocab from {len(vocab)} to {len(new_vocab)} words')
    # print(f'Removed: {(list(set(removed)))}')
    return result


def preprocess(docs: list[str], stem_words=True, limit=0) -> list[list[str]]:
    '''Apply all preprocessing steps to the given docs'''
    noteless = remove_notes(docs)
    just_words = remove_punctuation(noteless)
    normalized = normalize(just_words)
    tokens = tokenize(normalized)
    no_stopwords = remove_stopwords(tokens)

    if stem_words:
        documents = stem(no_stopwords)
    else:
        documents = lemmatize(no_stopwords)

    if limit > 0:
        documents = remove_rare_words(documents, limit=limit)

    return documents


if __name__ == "__main__":
    genres = ['rap', 'rock', 'pop']
    limit = '10k'
    filename = f'data/song_lyrics_reduced_{"_".join(genres)}_{limit}.csv'

    print("Loading csv")
    df = pd.read_csv(filename)
    documents = df['lyrics']
    print("Loaded succesfully")

    print("\nPreprocessing data")
    noteless = remove_notes(documents)
    just_words = remove_punctuation(noteless)
    normalized = normalize(just_words)
    tokens = tokenize(normalized)
    no_stopwords = remove_stopwords(tokens)

    print("Lemmatizing")
    lemmatized = lemmatize(no_stopwords)
    for document in lemmatized[:5]:
        print(' '.join(document))

    print("\nStemming")
    stemmed = stem(no_stopwords)
    for document in stemmed[:5]:
        print(' '.join(document))

    print("\nRemoving rare words (lemmatized)")
    no_rare_words = remove_rare_words(lemmatized, limit=3)
    print("\nRemoving rare words (stemmed)")
    no_rare_words = remove_rare_words(stemmed, limit=3)