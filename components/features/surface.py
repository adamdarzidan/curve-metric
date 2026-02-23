from ..data_module import SurfaceFeatures
from spacy.tokens.span import Span
from spacy.matcher import PhraseMatcher
from dataclasses import dataclass, fields

import json

class SurfaceDecoder:
   def __init__(self, CONNECTIVES_PATH = "data/ruleset.json"):
      try:
         with open(CONNECTIVES_PATH) as file:
            data = json.load(file)
            conns = data.get("connectives", {})

            self.CAUSAL_CONNECTIVES = conns.get("causal_connectives", [])
            self.TEMPORAL_CONNECTIVES = conns.get("temporal_connectives", [])
            self.LOGICAL_CONNECTIVES = conns.get("logical_connectives", [])
            self.ADDITIVE_CONNECTIVES = conns.get("additive_connectives", [])
            self.ADVERSATIVE_CONNECTIVES = conns.get("adversative_connectives", [])
            
      except FileNotFoundError:
         print(f"File not found: path {CONNECTIVES_PATH} does not exist")
      
   def extract_surface_features(self, span: Span) -> SurfaceFeatures:
      matcher = span.vocab
      surface_features = SurfaceFeatures()
      
      surface_features.sentence_length = len(span.text)
      surface_features.word_count = len(span)
      
      for token in span:
         pos = token.pos_
         text = token.text.lower().strip()
         if pos  in {"CCONJ", "SCONJ"}:
            if text in self.CAUSAL_CONNECTIVES:
               surface_features.causal_connectives += 1
            if text in self.ADDITIVE_CONNECTIVES:
               surface_features.additive_connectives += 1
            if text in self.ADVERSATIVE_CONNECTIVES:
               surface_features.adversative_connectives += 1
            if text in self.LOGICAL_CONNECTIVES:
               surface_features.logical_connectives += 1
            if text in self.TEMPORAL_CONNECTIVES:
               surface_features.temporal_connectives += 1   
            
      return surface_features
   
   def print_val(self, span):
      print("----------------SPAN---------------")
      print(span)
      print("----------------SURFACE---------------")
      t: SurfaceFeatures = self.extract_surface_features(span)
      for field in fields(SurfaceFeatures):
         print(f"{field.name}: {getattr(t, field.name)}")
      print("-------------------------------")