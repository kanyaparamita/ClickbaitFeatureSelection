import json
import pandas as pd
import spacy
from spacy.tokens import Doc
from pycorenlp import StanfordCoreNLP

# clickbaits tend to have longer headlines
def number_of_words(doc: Doc) -> int:
    return len(doc)

# Clickbaits tend to have smaller words; simpler words (short words more common in everyday english),
# and more common for word shortening (e.g. "Don't" instead of "do not")
def avg_length_of_word(doc: Doc) -> float:
    total_len = 0
    for token in doc:
        total_len += len(token)
    return total_len / len(doc)

# non-clickbaits use more descriptive words
def stopword_percentage(doc: Doc) -> float:
    count_sw = 0
    for token in doc:
        count_sw += token.is_stop
    return count_sw / len(doc)

# clickbait title uses number more often to list. non-clickbaits also include numbers but much rare and
# more meaningful (e.g. year, number of people dead)
# not a feature in the original paper
def has_number(doc: Doc) -> bool:
    for token in doc:
        if token.pos_ == 'NUM':
            return True
    return False

def has_determiner(doc: Doc) -> bool:
    for token in doc:
        if token.pos_ == 'DET':
            return True
    return False

def has_pronoun(doc: Doc) -> bool:
    for token in doc:
        if token.pos_ == 'PRON':
            return True
    return False

def has_comparative(doc: Doc) -> bool:
    for token in doc:
        if token.tag_ == 'JJR' or token.tag_ == 'RBR':
            return True
    return False

def has_superlative(doc: Doc) -> bool:
    for token in doc:
        if token.tag_ == 'JJS' or token.tag_ == 'RBS':
            return True
    return False

def sentiment_score(sentence: str):
    snlp = StanfordCoreNLP('http://localhost:9000')
    result = snlp.annotate(sentence,
        properties={
            'annotators': 'sentiment',
            'outputFormat': 'json',
            'timeout': 10000,
        })
    result_data = json.loads(result)  # Parse the JSON data
    return result_data["sentences"][0]["sentimentValue"]

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    nlp = spacy.load("en_core_web_lg")
    total_rows = df.shape[0]

    columns = ['headline', 'class', '#words','word length', 'stopword%', 'has number', 'has determiner', 'has pronoun',
               'has comparative', 'has superlative', 'has sup or comp', 'sentiment value']

    # statistics for clickbait data
    df_features = pd.DataFrame(index=range(total_rows), columns=columns)

    for index, row in df.iterrows():
        sentence = row["headline"].lower()
        doc = nlp(sentence)

        df_features["headline"][index] = sentence
        df_features["class"][index] = row["label"]
        df_features["#words"][index] = number_of_words(doc)
        df_features["word length"][index] = avg_length_of_word(doc)
        df_features["stopword%"][index] = stopword_percentage(doc)
        df_features["has number"][index] = int(has_number(doc))
        df_features["has determiner"][index] = int(has_determiner(doc))
        df_features["has pronoun"][index] = int(has_pronoun(doc))
        df_features["has comparative"][index] = int(has_comparative(doc))
        df_features["has superlative"][index] = int(has_superlative(doc))
        df_features["has sup or comp"][index] = int(has_comparative(doc) or has_superlative(doc))
        df_features["sentiment value"][index] = sentiment_score(sentence)

        print(index / total_rows, end="\r", flush=True)

    return df_features
