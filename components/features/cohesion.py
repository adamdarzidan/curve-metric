import numpy as np
from ..data_module import CohesionFeatures, DocumentFeatures


class CohesionDecoder:

    def __init__(self, doc_features: DocumentFeatures):
        self.doc = doc_features
        self.cache = doc_features.sentence_cache

        emb = doc_features.sentence_embeddings

        # L2 normalize embeddings safely
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        self.emb = np.divide(emb, norms, out=np.zeros_like(emb), where=norms != 0)
        # Store norms separately (useful signal)
        self.norms = norms.flatten()

    # -----------------------------
    # Stable similarity utilities
    # -----------------------------
    def _cosine(self, a, b):
        return float(np.dot(a, b))

    def _jaccard(self, a, b):
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union > 0 else 0.0

    def _ratio(self, num, den):
        return num / den if den > 0 else 0.0

    # -----------------------------
    # Feature extraction
    # -----------------------------
    def extract_cohesion_features(self, index: int):

        f = CohesionFeatures()

        if index == 0:
            return f

        prev = self.cache[index - 1]
        curr = self.cache[index]
        doc = self.doc
        emb = self.emb

        prev_lemmas = prev["lemmas"]
        curr_lemmas = curr["lemmas"]

        prev_nouns = prev["nouns"]
        curr_nouns = curr["nouns"]

        prev_args = prev["args"]
        curr_args = curr["args"]

        prev_verbs = prev["verbs"]
        curr_verbs = curr["verbs"]

        # -----------------------------
        # Lexical overlap 
        # -----------------------------
        if prev_lemmas:
            f.content_overlap_adjacent = self._jaccard(prev_lemmas, curr_lemmas)

        if doc.all_content_lemmas:
            f.content_overlap_all = self._jaccard(curr_lemmas, doc.all_content_lemmas)

        if prev_nouns:
            f.noun_overlap_adjacent = self._jaccard(prev_nouns, curr_nouns)

        if prev_args:
            f.argument_overlap_adjacent = self._jaccard(prev_args, curr_args)

        if doc.all_stems:
            f.stem_overlap_all = self._jaccard(curr_lemmas, doc.all_stems)

        if prev_verbs:
            f.verb_overlap_adjacent = self._jaccard(prev_verbs, curr_verbs)

        # -----------------------------
        # Embedding similarity 
        # -----------------------------
        curr_emb = emb[index]
        prev_emb = emb[index - 1]

        f.lsa_overlap_adjacent = self._cosine(curr_emb, prev_emb)
        # Embedding norm features (lost during normalization otherwise)
        f.embedding_norm = self.norms[index]
        f.embedding_norm_diff = self.norms[index] - self.norms[index - 1]

        if index > 1:
            prev_embs = emb[:index]
            sims = prev_embs @ curr_emb  # cosine similarity vector

            # Distributional embedding features (MUCH richer signal)
            f.lsa_overlap_mean = float(np.mean(sims))
            f.lsa_overlap_std = float(np.std(sims))
            f.lsa_overlap_max = float(np.max(sims))
            f.lsa_overlap_min = float(np.min(sims))

            # Novelty = how different from most similar past sentence
            f.lsa_novelty = 1.0 - f.lsa_overlap_max

            # Keep compatibility
            f.lsa_overlap_all = f.lsa_overlap_mean
            f.lsa_given_new = 1.0 - f.lsa_overlap_mean

            # Shift between local and global cohesion
            f.lsa_shift = f.lsa_overlap_adjacent - f.lsa_overlap_mean
        else:
            f.lsa_overlap_mean = f.lsa_overlap_adjacent
            f.lsa_overlap_std = 0.0
            f.lsa_overlap_max = f.lsa_overlap_adjacent
            f.lsa_overlap_min = f.lsa_overlap_adjacent
            f.lsa_novelty = 1.0 - f.lsa_overlap_adjacent

            f.lsa_overlap_all = f.lsa_overlap_adjacent
            f.lsa_given_new = 1.0 - f.lsa_overlap_adjacent
            f.lsa_shift = 0.0

        # -----------------------------
        # POS stability 
        # -----------------------------
        pos_prev = prev["pos"]
        pos_curr = curr["pos"]

        if pos_prev and pos_curr:
            overlap = sum(a == b for a, b in zip(pos_prev, pos_curr))
            f.pos_dissimilarity_prev = 1.0 - self._ratio(
                overlap, max(len(pos_prev), len(pos_curr))
            )
        else:
            f.pos_dissimilarity_prev = 0.0

        # -----------------------------
        # Word dissimilarity
        # -----------------------------
        if prev_lemmas:
            new_words = curr_lemmas - prev_lemmas
            f.word_dissimilarity_prev = self._ratio(len(new_words), len(prev_lemmas))

            # Stronger signal: proportion of entirely new vocabulary
            f.new_word_ratio = self._ratio(len(new_words), len(curr_lemmas))
        else:
            f.word_dissimilarity_prev = 0.0
            f.new_word_ratio = 0.0

        # -----------------------------
        # FIXED: Type-token ratio (was broken before)
        # -----------------------------
        curr_tokens = curr.get("tokens", curr["lemmas"])
        unique_tokens = len(set(curr_tokens))

        f.type_token_ratio = self._ratio(unique_tokens, len(curr_tokens))
        # Log-scaled length (helps stabilize variance)
        f.sentence_length = len(curr_tokens)
        f.sentence_length_log = np.log1p(len(curr_tokens))

        # -----------------------------
        # Document-level features (kept as global context)
        # -----------------------------
        f.lexical_diversity_all = doc.lexical_diversity_all
        f.lexical_diversity_verbs = doc.lexical_diversity_verbs

        return f