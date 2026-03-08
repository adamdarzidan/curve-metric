
from .linguistic_processor import LinguisticProcessor
from .data_module import *
import spacy
from spacy.tokens import Doc, Span


from .features.surface import SurfaceDecoder
from .features.syntax import SyntaxDecoder
from .features.lexical import LexicalDecoder
from .features.cohesion import CohesionDecoder
from .features.document import DocumentExtracter
from .data_module import DocumentProfile

class FeatureProfiler:
    
    def __init__(self, lp: LinguisticProcessor):
        self.lp = lp
        self.document_level_features = DocumentExtracter()
        self.lexical_decoder = LexicalDecoder()
        self.surface_decoder = SurfaceDecoder()
        self.syntax_decoder = SyntaxDecoder()
        self.cohesion_decoder = []
    
from dataclasses import fields
import numpy as np
from .linguistic_processor import LinguisticProcessor
from .data_module import *
import spacy
from spacy.tokens import Doc, Span

from .features.surface import SurfaceDecoder
from .features.syntax import SyntaxDecoder
from .features.lexical import LexicalDecoder
from .features.cohesion import CohesionDecoder
from .features.document import DocumentExtracter
from .data_module import DocumentProfile, FeatureStats, SentenceFeatures, SurfaceFeatures, LexicalFeatures, SyntaxFeatures, CohesionFeatures


class FeatureProfiler:
    
    def __init__(self, lp: LinguisticProcessor):
        self.lp = lp
        self.document_level_features = DocumentExtracter()
        self.lexical_decoder = LexicalDecoder()
        self.surface_decoder = SurfaceDecoder()
        self.syntax_decoder = SyntaxDecoder()
        self.cohesion_decoder = []

    def extract(self, text) -> DocumentProfile:
        doc: Doc = self.lp.process(text) if hasattr(self.lp, 'process') else self.lp(text)
        sentence_features: list[SentenceFeatures] = []
        self.cohesion_decoder = CohesionDecoder(self.document_level_features.extract(doc))
        prev_sent: Span | None = None

        for index, sent in enumerate(doc.sents):
            surface_feat = self.surface_decoder.extract_surface_features(sent)
            lexical_feat = self.lexical_decoder.extract_lexical_features(sent)
            syntax_feat = self.syntax_decoder.extract_syntax_features(sent)
            if prev_sent is None:
                cohesion_feat = CohesionFeatures()
            else:
                cohesion_feat = self.cohesion_decoder.extract_cohesion_features(index, prev_sent, sent)
            prev_sent = sent
            sentence_features.append(SentenceFeatures(index, sent.text, surface_feat, syntax_feat, lexical_feat, cohesion_feat))

        def make_stats(values):
            values = np.array(values) if values else np.array([0])
            return FeatureStats(
                avg=float(np.mean(values)),
                sd=float(np.std(values)),
                min=float(np.min(values)),
                max=float(np.max(values))
            )

        def aggregate(feature_name, cls_type):
            values = []
            for sf in sentence_features:
                feature_obj = getattr(sf, cls_type)
                val = getattr(feature_obj, feature_name, 0)
                values.append(val)
            return make_stats(values)

        # Build the DocumentProfile
        doc_profile = DocumentProfile(
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
            syntactic_similarity_prev=None,  # not used
            content_overlap_adjacent=aggregate("content_overlap_adjacent", "cohesion"),
            content_overlap_all=aggregate("content_overlap_all", "cohesion"),
            noun_overlap_adjacent=aggregate("noun_overlap_adjacent", "cohesion"),
            argument_overlap_adjacent=aggregate("argument_overlap_adjacent", "cohesion"),
            stem_overlap_all=aggregate("stem_overlap_all", "cohesion"),
            lsa_overlap_adjacent=aggregate("lsa_overlap_adjacent", "cohesion"),
            lsa_overlap_all=aggregate("lsa_overlap_all", "cohesion"),
            lsa_given_new=aggregate("lsa_given_new", "cohesion"),
            lsa_verb_overlap_adjacent=aggregate("lsa_verb_overlap_adjacent", "cohesion"),
            pos_dissimilarity_prev=aggregate("pos_dissimilarity_prev", "cohesion"),
            word_dissimilarity_prev=aggregate("word_dissimilarity_prev", "cohesion"),
            causal_cohesion=aggregate("causal_cohesion", "cohesion"),
            intentional_cohesion=aggregate("intentional_cohesion", "cohesion"),
            temporal_cohesion=aggregate("temporal_cohesion", "cohesion"),
            verb_overlap_adjacent=aggregate("verb_overlap_adjacent", "cohesion"),
            verb_tense_repetition=aggregate("verb_tense_repetition", "cohesion"),
            verb_aspect_repetition=aggregate("verb_aspect_repetition", "cohesion"),
            type_token_ratio=aggregate("type_token_ratio", "cohesion"),
            lexical_diversity_all=aggregate("lexical_diversity_all", "cohesion"),
            lexical_diversity_verbs=aggregate("lexical_diversity_verbs", "cohesion")
        )
        return doc_profile