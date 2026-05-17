# Libraries for nlp
import numpy as np
from spacy.tokens import Doc, Span
# Linguistic processor carrying nlp model
from .linguistic_processor import LinguisticProcessor
# Functors for extracting 
from .features.surface import SurfaceDecoder
from .features.syntax import SyntaxDecoder
from .features.lexical import LexicalDecoder
from .features.cohesion import CohesionDecoder
from .features.document import DocumentExtracter

# Templates for data array
from .data_module import DocumentProfile, FeatureStats, SentenceFeatures, CohesionFeatures

class FeatureExtractor:
    
    def __init__(self, lp: LinguisticProcessor):
        self.linguistic_processor = lp
        self.lexical_decoder = LexicalDecoder()
        self.surface_decoder = SurfaceDecoder()
        self.syntax_decoder = SyntaxDecoder()
        self.cohesion_decoder = CohesionDecoder()
        
        
    def extract(self, text) -> DocumentProfile:
        # Process text and have ready document to iterate over
        doc: Doc = self.lp.process(text)
        # Prepare data for iteration
        sentences = np.ndarray([sent.text for sent in doc])
        sentence_features = np.ndarray(dtype=SentenceFeatures)
        self.cohesion_decoder.initialize_document(doc)
        
        
        
        
        
        
        
        