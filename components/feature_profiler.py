
from .linguistic_processor import LinguisticProcessor
from .data_module import *
import spacy
from spacy.tokens import Doc, Span


from .features import surface
from .features import syntax
from .features.lexical import LexicalDecoder
from .features import cohesion

class FeatureProfiler:
    
    def __init__(self, lp: LinguisticProcessor):
        self.lp = lp
        self.lexical_decoder = LexicalDecoder()
    
    def extract(self, text) -> list[SentenceFeatures]:
        doc: Doc = self.lp.process(text) if hasattr(self.lp, 'process') else self.lp(text)
        sentence_features: list[SentenceFeatures] = []

        prev_sent: Span | None = None
        for index, sent in enumerate(doc.sents):
            surface_feature: SurfaceFeatures = surface.extract_surface_features(sent)
            syntax_feature: SyntaxFeatures = syntax.extract_syntax_features(sent)
            lexical_feature: LexicalFeatures = self.lexical_decoder.extract_lexical_features(sent)
            self.lexical_decoder.print_val(sent)
            cohesion_feature: CohesionFeatures = cohesion.extract_cohesion_features(sent, prev_sent) if prev_sent is not None else CohesionFeatures()
            prev_sent = sent

            
            sentence_features.append(SentenceFeatures(index, sent.text, surface_feature, syntax_feature, lexical_feature, cohesion_feature))
            