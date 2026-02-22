from spacy.tokens.span import Span
from ..data_module import CohesionFeatures

def extract_cohesion_features(span: Span, prev_sent: Span | None) -> CohesionFeatures:
    CohesionFeatures('hi')