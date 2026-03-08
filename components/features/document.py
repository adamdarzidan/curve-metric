from ..data_module import DocumentFeatures
from spacy.tokens import Doc
from sentence_transformers import SentenceTransformer


class DocumentExtracter:
    """
    Extracts structured document-level cohesion features.
    """

    def __init__(self, embedding_model_name="all-MiniLM-L12-v2"):
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def extract(self, doc: Doc) -> DocumentFeatures:

        sentences = list(doc.sents)

        all_content_lemmas = set()
        all_noun_lemmas = set()
        all_verb_lemmas = set()
        all_stems = set()
        all_argument_lemmas = set()

        total_tokens = []
        total_verbs = []

        for sent in sentences:
            for token in sent:
                if token.is_alpha:
                    lemma = token.lemma_.lower()
                    total_tokens.append(lemma)
                    all_stems.add(lemma)

                    if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}:
                        all_content_lemmas.add(lemma)
                    if token.pos_ == "NOUN":
                        all_noun_lemmas.add(lemma)
                    if token.pos_ == "VERB":
                        all_verb_lemmas.add(lemma)
                        total_verbs.append(lemma)
                    if token.dep_ in {"nsubj", "dobj"}:
                        all_argument_lemmas.add(lemma)
        lexical_diversity_all = (
            len(set(total_tokens)) / len(total_tokens)
            if total_tokens else 0.0
        )
        lexical_diversity_verbs = (
            len(set(total_verbs)) / len(total_verbs)
            if total_verbs else 0.0
        )

        sentence_texts = [sent.text for sent in sentences]
        
        sentence_embeddings = self.embedding_model.encode(sentence_texts)

        return DocumentFeatures(
            all_content_lemmas=all_content_lemmas,
            all_noun_lemmas=all_noun_lemmas,
            all_verb_lemmas=all_verb_lemmas,
            all_stems=all_stems,
            all_argument_lemmas=all_argument_lemmas,
            sentence_embeddings=sentence_embeddings,
            lexical_diversity_all=lexical_diversity_all,
            lexical_diversity_verbs=lexical_diversity_verbs
        )