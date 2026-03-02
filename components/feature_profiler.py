
from .linguistic_processor import LinguisticProcessor
from .data_module import *
import spacy
from spacy.tokens import Doc, Span


from .features.surface import SurfaceDecoder
from .features.syntax import SyntaxDecoder
from .features.lexical import LexicalDecoder
from .features.cohesion import CohesionDecoder
from .features.document import DocumentExtracter

class FeatureProfiler:
    
    def __init__(self, lp: LinguisticProcessor):
        self.lp = lp
        self.document_level_features = DocumentExtracter()
        self.lexical_decoder = LexicalDecoder()
        self.surface_decoder = SurfaceDecoder()
        self.syntax_decoder = SyntaxDecoder()
        self.cohesion_decoder = []
    
    def extract(self, text) -> list[SentenceFeatures]:
        doc: Doc = self.lp.process(text) if hasattr(self.lp, 'process') else self.lp(text)
        sentence_features: list[SentenceFeatures] = []
        self.cohesion_decoder = CohesionDecoder(self.document_level_features.extract(doc))

        prev_sent: Span | None = None
        for index, sent in enumerate(doc.sents):
            surface_feature: SurfaceFeatures = self.surface_decoder.extract_surface_features(sent)
            lexical_feature: LexicalFeatures = self.lexical_decoder.extract_lexical_features(sent)
            syntax_feature: SyntaxFeatures = self.syntax_decoder.extract_syntax_features(sent)
            cohesion_feature: CohesionFeatures
            if prev_sent == None:
                cohesion_feature = CohesionFeatures()
            else:
                cohesion_feature = self.cohesion_decoder.extract_cohesion_features(index, prev_sent, sent)
                
            self.lexical_decoder.print_val(sent)
            self.surface_decoder.print_val(sent)
            self.cohesion_decoder.print_features("COHESTION", cohesion_feature)
            
            prev_sent = sent
            
            sentence_features.append(SentenceFeatures(index, sent.text, surface_feature, syntax_feature, lexical_feature, cohesion_feature))
            