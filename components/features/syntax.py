from spacy.tokens.span import Span
from ..data_module import SyntaxFeatures

def extract_syntax_features(span: Span) -> SyntaxFeatures:
    SyntaxFeatures(1, 1, 1)