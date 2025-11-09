import pandas as pd
import numpy as np
from regex import regex as re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import FreqDist
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

from rid import RegressiveImageryDictionary, DEFAULT_RID_DICTIONARY, DEFAULT_RID_EXCLUSION_LIST

RID = RegressiveImageryDictionary()
RID.load_dictionary_from_string(DEFAULT_RID_DICTIONARY)
RID.load_exclusion_list_from_string(DEFAULT_RID_EXCLUSION_LIST)
CATEGORIES = ['PRIMARY', 'SECONDARY', 'EMOTIONS']


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

def get_rid_values(documents):
    results = []
    for document in documents:
        result = RID.analyze(document)
        counts = {category: 0 for category in CATEGORIES}
        for key, value in result.category_count.items():
            category = key.parent.name
            if key.parent.parent.name != 'root':
                category = key.parent.parent.name
            counts[category] += value

        highest = ('None', 0)
        for category in CATEGORIES:
            if counts[category] > highest[1]:
                highest = (category, counts[category])
        
        results.append(highest[0])

    return results

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


def remove_rare_words(docs: list[list[str]], limit=1, location='', debug=True, return_count=False) -> list[list[str]]:
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
        print('; '.join([f'{word}: {round(distribution.freq(word) * size)}' for word in distribution_words[:10]]))
        print('Least frequent:')
        print(' - '.join([f'{word}: {round(distribution.freq(word) * size)}' for word in distribution_words[-10:]]))

    result = []
    removed = []
    for doc in docs:
        new_doc = [word for word in doc if round(distribution.freq(word) * size) > limit]
        removed += [word for word in doc if round(distribution.freq(word) * size) <= limit]
        result.append(new_doc)
    new_vocab = [word for word in vocab if round(distribution.freq(word) * size) > limit]

    output = []
    for word in vocab:
        output.append([word, round(distribution.freq(word) * size), round(distribution.freq(word) * size) > limit])
    if location != '':
        write_out(output, ['word', 'occurence', 'included'], location)

    if debug:
        print(f'\nRemoved all words occuring {limit} or less times')
        print(f'Reduced vocab from {len(vocab)} to {len(new_vocab)} words')

    if not return_count:
        return result
    else:
        return [len(vocab), len(new_vocab), result]
    

def apply_quantiles(counts, quantiles):
    # Divide all entries into a quantile
    quantiled = []
    for count in counts:
        for quant in quantiles:
            # Find the correct quantile that this count falls in
            if count <= quant:
                quantiled.append(int(quant))
                break
        # If the found value is outside of the range, apply the highest quantile to it anyway
        if count > quantiles[-1]:
            quantiled.append(int(quantiles[-1]))
    return quantiled


def get_quantiles(counts, quants, debug=False):
    # Set up quantile range based on desired number of quantiles
    quant_range = [(i / quants) for i in range(1, quants)] + [1.0]
    quantiles = np.quantile(counts, quant_range)
    if debug: print(f"Found quantiles: {quantiles} for {quant_range} (range {min(counts)} - {max(counts)})")
    # Return both the found quantiles and the counts divided into those quantiles
    return quantiles, apply_quantiles(counts, quantiles)


def preprocess(docs: list[str], stem_words=True, limit=0, debug=False, return_count=False, 
               use_rid=0, line_quants=0, token_quants=0, tpl_quants=0, use_length=0) -> list[str]:
    '''Apply all preprocessing steps to the given docs'''
    if debug: print("Preprocessing data")
    
    # Remove notes
    if debug: print("Removing notes in [brackets]")
    noteless = remove_notes(docs)

    # Sophisticated feature: length. Only viable at 2 or above, since only 1 bucket is ambiguous.
    if line_quants > 1:
        if debug: print("Counting Lines")
        # Count number of lines per song
        line_counts = [len(document.split('\n')) for document in noteless]
        line_quantiles, quantiled_lines = get_quantiles(line_counts, line_quants, debug=debug)

    # Punctuation
    if debug: print("Removing punctuation")
    just_words = remove_punctuation(noteless)
    if debug: print("Normalizing")
    normalized = normalize(just_words)

    # Sophisticated feature: RID
    rid_values = get_rid_values(normalized)
    if debug and use_rid > 0: print(f'Assigned {len(rid_values)} RID values ({list(set(rid_values))})')

    if debug: print("Tokenizing")
    tokens = tokenize(normalized)

    # Sophisticated feature: length Only viable at 2 or above, since only 1 bucket is ambiguous.
    if token_quants > 1:
        if debug: print("Counting tokens")
        # Calculate quantiles, similar to line_quantiles
        token_counts = [len(document) for document in tokens]
        token_quantiles, quantiled_tokens = get_quantiles(token_counts, token_quants, debug=debug)
        if tpl_quants > 1 and line_quants > 1:
            # Calculate quantiles, similar to line_quantiles
            if debug: print("Counting tokens per line")
            tpl_counts = [int(token_counts[i] // line_counts[i]) for i in range(len(docs))]
            tpl_quantiles, quantiled_tpl = get_quantiles(tpl_counts, tpl_quants, debug=debug)

    # Remove stopwords
    if debug: print("Removing stopwords")
    no_stopwords = remove_stopwords(tokens)

    # Get roots of words
    if stem_words:
        if debug: print("Stemming")
        documents = stem(no_stopwords)
    else:
        if debug: print("Lemmatizing")
        documents = lemmatize(no_stopwords)

    # Remove infrequent words
    if limit > 0 or return_count:
        if debug: print("Removing rare words")
        documents = remove_rare_words(documents, limit=limit, debug=debug, return_count=return_count)

    # Append RID and Length values
    for i in range(len(documents)):
        for j in range(use_rid):
            documents[i].append(rid_values[i])

        if line_quants > 1:
            # Append the same value "use_length" times, to strengthen its effect
            for k in range(use_length):
                documents[i].append(f'LINE_QUANT_{quantiled_lines[i]}')
        if token_quants > 1:
            # Append the same value "use_length" times, to strengthen its effect
            for k in range(use_length):
                documents[i].append(f'TOKEN_QUANT_{quantiled_tokens[i]}')
            if tpl_quants > 1 and line_quants > 1:
                # Append the same value "use_length" times, to strengthen its effect
                for k in range(use_length):
                    documents[i].append(f'TPL_QUANT_{quantiled_tpl[i]}')

    if debug: print("Finished data preparation!")
    if not return_count:
        return [' '.join(doc) for doc in documents], [line_quantiles, token_quantiles, tpl_quantiles]
    else:
        old, total, documents = documents
        return old, total, documents


def preprocess_validation(docs: list[str], line_quantiles, token_quantiles, tpl_quantiles, use_rid, use_length) -> list[str]:
    noteless = remove_notes(docs)

    # Count number of lines per song
    line_counts = [len(document.split('\n')) for document in noteless]
    quantiled_lines = apply_quantiles(line_counts, line_quantiles)

    just_words = remove_punctuation(noteless)
    normalized = normalize(just_words)
    rid_values = get_rid_values(normalized)
    tokens = tokenize(normalized)

    # Count tokens per song
    token_counts = [len(document) for document in tokens]
    quantiled_tokens = apply_quantiles(token_counts, token_quantiles)
    # Count tokens per line
    tpl_counts = [int(token_counts[i] // line_counts[i]) for i in range(len(docs))]
    quantiled_tpl = apply_quantiles(tpl_counts, tpl_quantiles)

    no_stopwords = remove_stopwords(tokens)
    documents = stem(no_stopwords)
    # Rare words do not have to be removed here, since our vectorizer will only recognize words in our feature space anyway

    for i in range(len(documents)):
        for j in range(use_rid):
            documents[i].append(rid_values[i])
        for k in range(use_length):
            documents[i].append(f'LINE_QUANT_{quantiled_lines[i]}')
        for k in range(use_length):
            documents[i].append(f'TOKEN_QUANT_{quantiled_tokens[i]}')
        for k in range(use_length):
            documents[i].append(f'TPL_QUANT_{quantiled_tpl[i]}')

    return [' '.join(doc) for doc in documents]


if __name__ == "__main__":
    genres = ['rap', 'rock', 'pop']
    limit = '1000'
    filename = f'data/song_lyrics_reduced_{"_".join(genres)}_{limit}.csv'

    print("Loading csv")
    df = pd.read_csv(filename)
    documents = df['lyrics']
    print("Loaded succesfully")
    preprocess(documents, debug=True)