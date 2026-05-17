from spacy.tokens.span import Span
from ..data_module import SyntaxFeatures

class SyntaxDecoder:
        
    def get_dependency_graph(self, token, depth):
        if token.n_lefts + token.n_rights == 0:
            return depth
        return max(self.get_dependency_graph(child, depth + 1) for child in token.children)

        
    def extract_syntax_features(self, span: Span):
        syntax_features = SyntaxFeatures()
        
        passive_construct = 0
        dependency_depth = self.get_dependency_graph(span.root, 0)
        
        words_before_main_verb = sum(
            1 for token in span
            if token.i < span.root.i and not token.is_punct and not token.is_space
        )

        noun_chunks = list(span.noun_chunks)
        noun_phrase_modifiers = 0
        
        for token in span:
            # Check for passive construct
            if token.dep_ in {"nsubjpass", "auxpass"}:
                passive_construct += 1

        modifier_deps = {"amod", "compound", "det", "nummod", "poss", "acl", "relcl", "appos"}
        for chunk in noun_chunks:
            noun_phrase_modifiers += sum(1 for token in chunk if token.dep_ in modifier_deps)
        
        syntax_features.dependency_depth = dependency_depth
        syntax_features.modifiers_per_np = noun_phrase_modifiers / max(len(noun_chunks), 1)
        syntax_features.passive_constructions = passive_construct
        syntax_features.words_before_main_verb = words_before_main_verb
        
        return syntax_features
        
    
        