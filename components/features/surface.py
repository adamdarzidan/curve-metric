from ..data_module import SurfaceFeatures
from spacy.tokens.span import Span

def extract_surface_features(span: Span) -> SurfaceFeatures:
   return SurfaceFeatures("he") 