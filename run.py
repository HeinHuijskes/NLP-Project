from DataPrep import preprocess
import pandas as pd


if __name__ == "__main__":
    file_name = 'data/song_lyrics_reduced_rap_rock_pop_1000.csv'
    data = pd.read_csv(file_name)
    result = preprocess(data)