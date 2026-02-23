from spacy.tokens.span import Span
from spacy.tokens import Token
from ..data_module import SyntaxFeatures

class SyntaxDecoder:
    def __init__(self):
        print("fill")
        
    def get_dependency_graph(self, token, depth):
        if token.n_lefts + token.n_rights == 0:
            return depth
        return max(self.get_dependency_graph(child, depth + 1) for child in token.children)

        
    def extract_syntax_features(self, span: Span):
        syntax_features = SyntaxFeatures()
        
        passive_construct = 0
        dependency_depth = self.get_dependency_graph(span.root, 0)
        
        words_before_main_verb = span.root.i
        
        for token in span:
            # Check for passive construct
            if token in ["nsubjpass", "auxpass"]:
                passive_construct += 1
        
        syntax_features.dependency_depth = dependency_depth
        syntax_features.passive_constructions = passive_construct
        syntax_features.words_before_main_verb = words_before_main_verb
        
        return syntax_features
        
    
        