from .data_module import SentenceFeatures
import spacy
from spacy.tokens import Doc
from spacy_syllables import SpacySyllables

class LinguisticProcessor:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)
        self.nlp.add_pipe("syllables", after="tagger")
        
    def process(self, text) -> Doc:
        return self.nlp(text)

    def __call__(self, text) -> Doc:
        return self.process(text)