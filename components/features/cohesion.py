from spacy.tokens import Span
from ..data_module import CohesionFeatures, DocumentFeatures
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import fields

class CohesionDecoder:
    def __init__(self, doc_features: DocumentFeatures):
        self.document_features: DocumentFeatures = doc_features

    def extract_cohesion_features(self, index: int, prev_sent: Span | None, sent: Span) -> CohesionFeatures:
        cohesion_features = CohesionFeatures()
        
        if prev_sent is None:
            # First sentence has no previous context → return default zeros
            return cohesion_features

        # Lexical / argument overlap
        # Content words
        content_prev = {t.lemma_.lower() for t in prev_sent if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}}
        content_curr = {t.lemma_.lower() for t in sent if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}}
        if content_prev:
            cohesion_features.content_overlap_adjacent = len(content_prev & content_curr) / len(content_prev)

        # Noun overlap
        nouns_prev = {t.lemma_.lower() for t in prev_sent if t.pos_ == "NOUN"}
        nouns_curr = {t.lemma_.lower() for t in sent if t.pos_ == "NOUN"}
        if nouns_prev:
            cohesion_features.noun_overlap_adjacent = len(nouns_prev & nouns_curr) / len(nouns_prev)

        # Argument overlap (subjects/objects)
        args_prev = {t.lemma_.lower() for t in prev_sent if t.dep_ in {"nsubj", "dobj"}}
        args_curr = {t.lemma_.lower() for t in sent if t.dep_ in {"nsubj", "dobj"}}
        if args_prev:
            cohesion_features.argument_overlap_adjacent = len(args_prev & args_curr) / len(args_prev)

        # Stem overlap (document-level)
        if self.document_features.all_stems:
            stems_curr = {t.lemma_.lower() for t in sent if t.is_alpha}
            cohesion_features.stem_overlap_all = len(stems_curr & self.document_features.all_stems) / len(self.document_features.all_stems)

        # Semantic / embeddings
        embedding_sent = self.document_features.sentence_embeddings[index].reshape(1, -1)
        embedding_prev_sent = self.document_features.sentence_embeddings[index - 1].reshape(1, -1)

        # LSA / embedding similarity
        cohesion_features.lsa_overlap_adjacent = float(cosine_similarity(embedding_sent, embedding_prev_sent)[0][0])

        # LSA overlap all previous
        if index > 1:
            prev_embeddings = self.document_features.sentence_embeddings[:index]
            similarities = cosine_similarity(embedding_sent, prev_embeddings)
            cohesion_features.lsa_overlap_all = float(similarities.mean())
            cohesion_features.lsa_given_new = 1.0 - float(similarities.max())
        else:
            cohesion_features.lsa_overlap_all = cohesion_features.lsa_overlap_adjacent
            cohesion_features.lsa_given_new = 1.0 - cohesion_features.lsa_overlap_adjacent

        # LSA verb overlap (semantic similarity on verbs)
        verbs_prev = [t.lemma_ for t in prev_sent if t.pos_ == "VERB"]
        verbs_curr = [t.lemma_ for t in sent if t.pos_ == "VERB"]
        if verbs_prev and verbs_curr:
            # approximate verb semantic overlap using sentence embeddings already computed
            cohesion_features.lsa_verb_overlap_adjacent = float(
                cosine_similarity(embedding_sent, embedding_prev_sent)[0][0]
            )

        # POS and lexical dissimilarity
        pos_prev = [t.pos_ for t in prev_sent]
        pos_curr = [t.pos_ for t in sent]
        if pos_prev:
            overlap = sum(1 for a, b in zip(pos_prev, pos_curr) if a == b)
            cohesion_features.pos_dissimilarity_prev = 1.0 - (overlap / max(len(pos_prev), len(pos_curr)))

        words_prev = {t.lemma_.lower() for t in prev_sent if t.is_alpha}
        words_curr = {t.lemma_.lower() for t in sent if t.is_alpha}
        if words_prev:
            cohesion_features.word_dissimilarity_prev = len(words_curr - words_prev) / len(words_prev)

        # Causal / temporal / intentional cohesion
        causal_connectives = {"because", "therefore", "so", "thus", "hence"}
        temporal_connectives = {"then", "next", "after", "before", "subsequently"}
        intentional_verbs = {"aim", "plan", "intend", "try"}

        sent_tokens = [t.text.lower() for t in sent]

        # Simple counts normalized by sentence length
        length = max(len(sent_tokens), 1)
        cohesion_features.causal_cohesion = sum(1 for t in sent_tokens if t in causal_connectives) / length
        cohesion_features.temporal_cohesion = sum(1 for t in sent_tokens if t in temporal_connectives) / length
        cohesion_features.intentional_cohesion = sum(1 for t in sent_tokens if t in intentional_verbs) / length

        # Verb overlap / tense / aspect
        verbs_prev_tokens = [t for t in prev_sent if t.pos_ == "VERB"]
        verbs_curr_tokens = [t for t in sent if t.pos_ == "VERB"]

        # Overlap
        if verbs_prev_tokens:
            overlap = set(t.lemma_ for t in verbs_prev_tokens) & set(t.lemma_ for t in verbs_curr_tokens)
            cohesion_features.verb_overlap_adjacent = len(overlap) / len(verbs_prev_tokens)

            # Tense repetition
            tense_match = sum(1 for t_curr, t_prev in zip(verbs_curr_tokens, verbs_prev_tokens)
                              if t_curr.morph.get("Tense") == t_prev.morph.get("Tense"))
            cohesion_features.verb_tense_repetition = tense_match / len(verbs_prev_tokens)

            # Aspect repetition
            aspect_match = sum(1 for t_curr, t_prev in zip(verbs_curr_tokens, verbs_prev_tokens)
                               if t_curr.morph.get("Aspect") == t_prev.morph.get("Aspect"))
            cohesion_features.verb_aspect_repetition = aspect_match / len(verbs_prev_tokens)

        # Lexical diversity / type-token ratio
        tokens = [t.lemma_.lower() for t in sent if t.is_alpha]
        unique_tokens = set(tokens)
        cohesion_features.type_token_ratio = len(unique_tokens) / max(len(tokens), 1)

        # Document-level diversity
        cohesion_features.lexical_diversity_all = self.document_features.lexical_diversity_all
        cohesion_features.lexical_diversity_verbs = self.document_features.lexical_diversity_verbs

        return cohesion_features
    
    from dataclasses import fields

    def print_features(self, name: str, features):
        """
        Print all fields and values of a dataclass with a header.

        Args:
            name (str): Name to display for the features (e.g., "SurfaceFeatures").
            features: A dataclass instance (SurfaceFeatures, LexicalFeatures, etc.)
        """
        print(f"----------------{name.upper()}---------------")
        for field in fields(features):
            value = getattr(features, field.name)
            print(f"{field.name}: {value}")
        print("-------------------------------")