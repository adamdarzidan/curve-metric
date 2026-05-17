from dataclasses import fields
import numpy as np
from .linguistic_processor import LinguisticProcessor
from .data_module import DocumentProfile, FeatureStats, SentenceFeatures, CohesionFeatures
from spacy.tokens import Doc

from .features.surface import SurfaceDecoder
from .features.syntax import SyntaxDecoder
from .features.lexical import LexicalDecoder
from .features.cohesion import CohesionDecoder
from .features.document import DocumentExtracter

from concurrent.futures import ThreadPoolExecutor


class FeatureProfiler:

    def __init__(self, lp: LinguisticProcessor):
        self.lp = lp
        self.document_extractor = DocumentExtracter()

        self.lexical_decoder = LexicalDecoder()
        self.surface_decoder = SurfaceDecoder()
        self.syntax_decoder = SyntaxDecoder()

    def extract(self, text) -> DocumentProfile:

        doc: Doc = self.lp.process(text) if hasattr(self.lp, "process") else self.lp(text)
        sentences = list(doc.sents)

        doc_features = self.document_extractor.extract(doc)
        cohesion_decoder = CohesionDecoder(doc_features)

        # local bindings for speed
        surface_fn = self.surface_decoder.extract_surface_features
        lexical_fn = self.lexical_decoder.extract_lexical_features
        syntax_fn = self.syntax_decoder.extract_syntax_features

        n = len(sentences)
        
        sentence_features = [None] * n

        for idx, sent in enumerate(sentences):
            surface = surface_fn(sent)
            syntax = syntax_fn(sent)
            lexical = lexical_fn(sent)
            cache_entry = doc_features.sentence_cache[idx]
            cache_entry["token_count"] = surface.word_count
            cache_entry["causal_connectives"] = surface.causal_connectives
            cache_entry["temporal_connectives"] = surface.temporal_connectives
            cache_entry["logical_connectives"] = surface.logical_connectives
            cache_entry["additive_connectives"] = surface.additive_connectives
            cache_entry["adversative_connectives"] = surface.adversative_connectives
            cache_entry["causal_verbs"] = lexical.causal_verbs
            cache_entry["intentional_actions"] = lexical.intentional_actions
            verb_tenses = set()
            verb_aspects = set()
            for token in sent:
                if token.pos_ == "VERB":
                    tense = token.morph.get("Tense")
                    aspect = token.morph.get("Aspect")
                    if tense:
                        verb_tenses.update(tense)
                    if aspect:
                        verb_aspects.update(aspect)
            cache_entry["verb_tenses"] = verb_tenses
            cache_entry["verb_aspects"] = verb_aspects
            sentence_features[idx] = SentenceFeatures(idx, sent.text, surface, syntax, lexical, cohesion_decoder.extract_cohesion_features(idx))
        


        def make_stats(values):

            if values is None or len(values) == 0:
                return FeatureStats(0.0, 0.0, 0.0, 0.0)

            arr = np.asarray(values, dtype=np.float32)

            if arr.size == 0:
                return FeatureStats(0.0, 0.0, 0.0, 0.0)

            return FeatureStats(
                avg=float(arr.mean()),
                sd=float(arr.std()),
                min=float(arr.min()),
                max=float(arr.max()),
            )

        def collect_values(feature_name, attr):
            values = []
            for sf in sentence_features:
                if sf is None:
                    values.append(0)
                    continue

                obj = getattr(sf, attr, None)
                if obj is None:
                    values.append(0)
                    continue

                values.append(getattr(obj, feature_name, 0))

            return values

        def aggregate(feature_name, attr):
            return make_stats(collect_values(feature_name, attr))

        return DocumentProfile(
            nouns=aggregate("nouns", "lexical"),
            verbs=aggregate("verbs", "lexical"),
            adjectives=aggregate("adjectives", "lexical"),
            adverbs=aggregate("adverbs", "lexical"),
            pronouns=aggregate("pronouns", "lexical"),
            first_person_pronouns=aggregate("first_person_pronouns", "lexical"),
            third_person_pronouns=aggregate("third_person_pronouns", "lexical"),
            avg_syllables_per_word=aggregate("avg_syllables_per_word", "lexical"),
            avg_word_frequency_log=aggregate("avg_word_frequency_log", "lexical"),
            avg_content_word_frequency_log=aggregate("avg_content_word_frequency_log", "lexical"),
            avg_age_of_acquisition=aggregate("avg_age_of_acquisition", "lexical"),
            avg_concreteness=aggregate("avg_concreteness", "lexical"),
            avg_imagery=aggregate("avg_imagery", "lexical"),
            avg_familiarity=aggregate("avg_familiarity", "lexical"),
            avg_polysemy=aggregate("avg_polysemy", "lexical"),
            negations=aggregate("negations", "lexical"),
            causal_verbs=aggregate("causal_verbs", "lexical"),
            intentional_actions=aggregate("intentional_actions", "lexical"),

            word_count=aggregate("word_count", "surface"),
            sentence_length=aggregate("sentence_length", "surface"),
            function_to_content_ratio=aggregate("function_to_content_ratio", "surface"),
            connectives_total=aggregate("connectives_total", "surface"),
            causal_connectives=aggregate("causal_connectives", "surface"),
            temporal_connectives=aggregate("temporal_connectives", "surface"),
            logical_connectives=aggregate("logical_connectives", "surface"),
            additive_connectives=aggregate("additive_connectives", "surface"),
            adversative_connectives=aggregate("adversative_connectives", "surface"),

            dependency_depth=aggregate("dependency_depth", "syntax"),
            modifiers_per_np=aggregate("modifiers_per_np", "syntax"),
            words_before_main_verb=aggregate("words_before_main_verb", "syntax"),
            passive_constructions=aggregate("passive_constructions", "syntax"),

            content_overlap_adjacent=aggregate("content_overlap_adjacent", "cohesion"),
            content_overlap_all=aggregate("content_overlap_all", "cohesion"),
            noun_overlap_adjacent=aggregate("noun_overlap_adjacent", "cohesion"),
            argument_overlap_adjacent=aggregate("argument_overlap_adjacent", "cohesion"),
            stem_overlap_all=aggregate("stem_overlap_all", "cohesion"),
            lsa_overlap_adjacent=aggregate("lsa_overlap_adjacent", "cohesion"),
            lsa_overlap_all=aggregate("lsa_overlap_all", "cohesion"),
            lsa_given_new=aggregate("lsa_given_new", "cohesion"),
            lsa_overlap_mean=aggregate("lsa_overlap_mean", "cohesion"),
            lsa_overlap_std=aggregate("lsa_overlap_std", "cohesion"),
            lsa_overlap_max=aggregate("lsa_overlap_max", "cohesion"),
            lsa_overlap_min=aggregate("lsa_overlap_min", "cohesion"),
            lsa_novelty=aggregate("lsa_novelty", "cohesion"),
            lsa_shift=aggregate("lsa_shift", "cohesion"),
            # lsa_verb_overlap_adjacent=aggregate("lsa_verb_overlap_adjacent", "cohesion"),
            pos_dissimilarity_prev=aggregate("pos_dissimilarity_prev", "cohesion"),
            word_dissimilarity_prev=aggregate("word_dissimilarity_prev", "cohesion"),
            new_word_ratio=aggregate("new_word_ratio", "cohesion"),
            verb_overlap_adjacent=aggregate("verb_overlap_adjacent", "cohesion"),
            embedding_norm=aggregate("embedding_norm", "cohesion"),
            embedding_norm_diff=aggregate("embedding_norm_diff", "cohesion"),
            # verb_tense_repetition=aggregate("verb_tense_repetition", "cohesion"),
            # verb_aspect_repetition=aggregate("verb_aspect_repetition", "cohesion"),
            type_token_ratio=aggregate("type_token_ratio", "cohesion"),
            lexical_diversity_all=aggregate("lexical_diversity_all", "cohesion"),
            lexical_diversity_verbs=aggregate("lexical_diversity_verbs", "cohesion"),
            sentence_length_cohesion=aggregate("sentence_length", "cohesion"),
            sentence_length_log=aggregate("sentence_length_log", "cohesion"),
        )