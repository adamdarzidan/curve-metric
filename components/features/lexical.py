from spacy.tokens.span import Span
from ..data_module import LexicalFeatures
import nltk
from wordfreq import zipf_frequency
from nltk.corpus import wordnet as wn
import os

from dataclasses import fields

import polars as pl
import json
from util import avg

from huggingface_hub import login
from config import HF_TOKEN

_HF_LOGGED_IN = False


def ensure_hf_login():
    global _HF_LOGGED_IN
    if _HF_LOGGED_IN:
        return

    token = os.getenv("HF_TOKEN")
    if token:
        login(token)
        _HF_LOGGED_IN = True
        
class LexicalDecoder:
    def __init__(self, config_token = HF_TOKEN, RULESET_PATH = "data/ruleset.json"):
        # Login into hugging face
        try:
            ensure_hf_login()
        except ConnectionRefusedError:
            print("Couldn't login. Token provided not valid")
            
        self.FIRST_PERSON_PRON = set()
        self.CAUSAL_VERBS = {"cause", "produce", "create", "lead", "result", "trigger", "make"}
        self.INTENTIONAL_ACTIONS = {"aim", "plan", "intend", "seek", "try", "attempt", "choose"}
        self.wordnet_available = True
        # Load datasets and dictionaries
        try:
            with open(RULESET_PATH) as file:
                data = json.load(file)
            self.FIRST_PERSON_PRON = {word.strip().lower() for word in data["first_person"]}
        except FileNotFoundError:
            print(f"File not found: update \"{RULESET_PATH}\" import const")
        
        # Ensure WordNet is available and actually usable
        try:
            nltk.download('wordnet', quiet=True)
            from nltk.corpus import wordnet as wn
            wn.ensure_loaded()  # force load to catch path issues
            self._polysemy_cache = {}
        except Exception:
            self.wordnet_available = False
            print("Warning: WordNet not usable; avg_polysemy will default to 0.")
        self.CONCRETENESS_DICT = {}
        self.FAMILIARITY_DICT = {}
        self.IMAGERY_DICT = {} 
        self.AOA_DICT = {}
        
        # Hugging face database for the four data typs
        try:
            df = pl.read_csv('hf://datasets/StephanAkkerman/MRC-psycholinguistic-database/mrc_psycholinguistic_database.csv', null_values=["NA"], try_parse_dates=False)
            for row in df.iter_rows(named=True):
                word = row["Word"]
                aoa = row.get("Age of Acquisition Rating")
                conc = row.get("Concreteness")
                fam = row.get("Familiarity")
                imagery = row.get("Imageability")

                if None in [aoa, conc, fam, imagery, word]:
                    continue
                
                word = word.strip().lower()
                
                self.AOA_DICT[word] = float(aoa)
                self.CONCRETENESS_DICT[word] = float(conc)
                self.FAMILIARITY_DICT[word] = float(fam)
                self.IMAGERY_DICT[word] = float(imagery)
        except FileNotFoundError:
            print("File not found. Change path.")

    # Extracting lexical features
    def extract_lexical_features(self, span: Span) -> LexicalFeatures:
        if span.__len__ == 0:
            return LexicalFeatures()
        
        # initalie all vars
        
        lexical_features = LexicalFeatures()
        span_len = len(span)
        
        # Frequency of each word in the English corpus
        total_word_freq = 0
        
        # Manages content word frequency: i.e. the frequency of words that arent stop words
        total_content_word_freq = 0
        count_content_word = 0
        
        syllables_count = 0
        
        t_word_concreteness = 0.0
        t_word_concreteness_defined = 0
        
        t_aoa = 0.0
        t_aoa_defined = 0
        
        t_imagery = 0.0
        t_imagery_defined = 0
        
        t_familiarity = 0.0
        t_familiarity_defined = 0

        t_polysemy = 0.0
        t_polysemy_defined = 0



        # Start iterating through every token in span
        for token in span:
            if token.pos_ == "PUNCT" or token.is_space:
                span_len -= 1
                continue

            token_text = token.text.lower()
            token_lemma = token.lemma_.lower()

            # Add to total word freq
            word_freq = zipf_frequency(token_text, "en")
            total_word_freq += word_freq
            
            # Extracting POS tags
            match token.pos_:
                case "NOUN" | "PROPN":
                    lexical_features.nouns += 1
                    total_content_word_freq += word_freq
                    count_content_word += 1
                case "VERB":
                    lexical_features.verbs += 1
                    total_content_word_freq+= word_freq
                    count_content_word += 1
                case "ADJ":
                    total_content_word_freq+= word_freq
                    count_content_word += 1
                    lexical_features.adjectives += 1
                case "ADV":
                    total_content_word_freq += word_freq
                    count_content_word += 1
                    lexical_features.adverbs += 1
                case "PRON":
                    lexical_features.pronouns += 1
                    if token_text in self.FIRST_PERSON_PRON:
                        lexical_features.first_person_pronouns += 1
                    else:
                        lexical_features.third_person_pronouns += 1

            if token.pos_ == "VERB":
                if token_lemma in self.CAUSAL_VERBS:
                    lexical_features.causal_verbs += 1
                if token_lemma in self.INTENTIONAL_ACTIONS:
                    lexical_features.intentional_actions += 1
                        
            # Check for if the word involves negation
            if token.dep_ == "neg":
                lexical_features.negations += 1            
            
            # Keep sum of syllables per word to average out in end
            syllables_count += token._.syllables_count or 0 # type: ignore
            
            # Calculate word concretness, imagery, age of aquisition, and familiarity
            word_concreteness = self.CONCRETENESS_DICT.get(token_lemma) or None
            if word_concreteness is not None:
                t_word_concreteness += word_concreteness
                t_word_concreteness_defined += 1
            
            age_of_aqui = self.AOA_DICT.get(token_lemma) or None
            if age_of_aqui is not None:
                t_aoa += age_of_aqui
                t_aoa_defined += 1
                
            familiarity = self.FAMILIARITY_DICT.get(token_lemma) or None
            if familiarity is not None:
                t_familiarity += familiarity
                t_familiarity_defined += 1
            
            imagery = self.IMAGERY_DICT.get(token_lemma) or None
            if imagery is not None:
                t_imagery += imagery
                t_imagery_defined += 1

            if self.wordnet_available:
                try:
                    if not hasattr(self, "_polysemy_cache"):
                        self._polysemy_cache = {}

                    if token_lemma in self._polysemy_cache:
                        poly = self._polysemy_cache[token_lemma]
                    else:
                        synsets = wn.synsets(token_lemma)
                        poly = len(synsets)
                        self._polysemy_cache[token_lemma] = poly

                    if poly > 0:
                        t_polysemy += poly
                        t_polysemy_defined += 1

                except Exception:
                    # WordNet can crash under multiprocessing/threading (zipfile bug)
                    self.wordnet_available = False
            
            
        # Final calculations to return        
        
        lexical_features.avg_syllables_per_word = avg(syllables_count, span_len)
        lexical_features.avg_word_frequency_log = avg(total_word_freq, span_len)
        lexical_features.avg_content_word_frequency_log = avg(total_content_word_freq, count_content_word)
        
        lexical_features.avg_concreteness = avg(t_word_concreteness, t_word_concreteness_defined)
        lexical_features.avg_age_of_acquisition = avg(t_aoa, t_aoa_defined)
        lexical_features.avg_imagery = avg(t_imagery, t_imagery_defined)
        lexical_features.avg_familiarity = avg(t_familiarity, t_familiarity_defined)
        lexical_features.avg_polysemy = avg(t_polysemy, t_polysemy_defined)
        
                
        return lexical_features
    
    
    def print_val(self, span):
        print("----------------SPAN---------------")
        print(span)
        print("----------------LEXICAL---------------")
        t: LexicalFeatures = self.extract_lexical_features(span)
        for field in fields(LexicalFeatures):
            print(f"{field.name}: {getattr(t, field.name)}")
        print("-------------------------------")