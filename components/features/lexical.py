import pandas
import spacy

def extract_lexical_features(sent: str):
    nlp = spacy.load('en')
    doc = nlp(sent)
