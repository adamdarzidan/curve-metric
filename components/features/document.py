from ..data_module import DocumentFeatures
from spacy.tokens import Doc
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import logging
from config import FeatureConfig


class DocumentExtracter:

    def __init__(self, embedding_model_name="all-MiniLM-L12-v2"):
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            device=self.device
        )

        self.config = FeatureConfig()

    def extract(self, doc: Doc, batch_size: int = 32) -> DocumentFeatures:

        content = set()
        nouns = set()
        verbs = set()
        args = set()
        stems = set()

        total_tokens = []
        total_verbs = []

        cfg = self.config

        sentence_cache = []

        for token in doc:   
            if not token.is_alpha:
                continue

            lemma = token.lemma_.lower()
            pos = token.pos_
            dep = token.dep_

            total_tokens.append(lemma)
            stems.add(lemma)

            if pos in cfg.CONTENT_LEMMAS:
                content.add(lemma)

            if pos == "NOUN":
                nouns.add(lemma)

            elif pos == "VERB":
                verbs.add(lemma)
                total_verbs.append(lemma)

            if dep in cfg.ARG_LEMMAS:
                args.add(lemma)

        n_tokens = len(total_tokens)

        lexical_diversity_all = len(set(total_tokens)) / n_tokens if n_tokens else 0.0

        n_verbs = len(total_verbs)
        lexical_diversity_verbs = len(set(total_verbs)) / n_verbs if n_verbs else 0.0

        sentences = []
        sentence_cache = []

        for sent in doc.sents:
            sentences.append(sent.text)

            lemmas = set()
            sent_nouns = set()
            sent_verbs = set()
            sent_args = set()
            pos_list = []

            for token in sent:
                if not token.is_alpha:
                    continue

                lemma = token.lemma_.lower()
                pos = token.pos_
                dep = token.dep_

                lemmas.add(lemma)
                pos_list.append(pos)

                if pos == "NOUN":
                    sent_nouns.add(lemma)
                elif pos == "VERB":
                    sent_verbs.add(lemma)

                if dep in cfg.ARG_LEMMAS:
                    sent_args.add(lemma)

            sentence_cache.append({
                "lemmas": lemmas,
                "nouns": sent_nouns,
                "verbs": sent_verbs,
                "args": sent_args,
                "pos": pos_list
            })

        sentence_embeddings = self.embedding_model.encode(
            sentences,
            batch_size=batch_size,
            device=self.device,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return DocumentFeatures(
            all_content_lemmas=content,
            all_noun_lemmas=nouns,
            all_verb_lemmas=verbs,
            all_stems=stems,
            all_argument_lemmas=args,
            sentence_cache=sentence_cache,
            sentence_embeddings=sentence_embeddings,
            lexical_diversity_all=lexical_diversity_all,
            lexical_diversity_verbs=lexical_diversity_verbs
        )