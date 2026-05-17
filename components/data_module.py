from dataclasses import dataclass, field
import numpy as np


@dataclass
class FeatureStats:
    avg: float
    sd: float
    min: float | int
    max: float | int

@dataclass
class LexicalFeatures:
    # POS counts
    nouns: int = 0
    verbs: int = 0
    adjectives: int = 0
    adverbs: int = 0
    pronouns: int = 0
    first_person_pronouns: int = 0
    third_person_pronouns: int = 0

    # Per word properties
    avg_syllables_per_word: float = 0.0
    avg_word_frequency_log: float = 0.0
    avg_content_word_frequency_log: float = 0.0
    avg_age_of_acquisition: float = 0.0 
    avg_concreteness: float = 0.0 
    avg_imagery: float = 0.0 
    avg_familiarity: float = 0.0 
    avg_polysemy: float = 0.0 #

    # semantic flags
    negations: int = 0 
    causal_verbs: int = 0 #
    intentional_actions: int = 0 #
    
@dataclass
class SurfaceFeatures:
    word_count: int = 0
    sentence_length: int = 0
    function_to_content_ratio: float = 0.0 # to be removed

    # connectives
    connectives_total: int = 0
    causal_connectives: int = 0
    temporal_connectives: int = 0
    logical_connectives: int = 0
    additive_connectives: int = 0
    adversative_connectives: int = 0
    
    
@dataclass
class SyntaxFeatures:
    dependency_depth: int = 0
    modifiers_per_np: float = 0.0 #
    words_before_main_verb: int = 0
    passive_constructions: int = 0
    
@dataclass
class CohesionFeatures:
    content_overlap_adjacent: float = 0.0
    content_overlap_all: float = 0.0
    noun_overlap_adjacent: float = 0.0
    argument_overlap_adjacent: float = 0.0
    stem_overlap_all: float = 0.0

    # -----------------------------
    # Embedding-based cohesion (ENHANCED)
    # -----------------------------
    lsa_overlap_adjacent: float = 0.0
    lsa_overlap_all: float = 0.0
    lsa_given_new: float = 0.0

    # NEW: distributional embedding features
    lsa_overlap_mean: float = 0.0
    lsa_overlap_std: float = 0.0
    lsa_overlap_max: float = 0.0
    lsa_overlap_min: float = 0.0

    # NEW: semantic structure
    lsa_novelty: float = 0.0
    lsa_shift: float = 0.0

    lsa_verb_overlap_adjacent: float = 0.0

    # -----------------------------
    # Embedding norm features (NEW)
    # -----------------------------
    embedding_norm: float = 0.0
    embedding_norm_diff: float = 0.0

    # -----------------------------
    # Structural / syntactic
    # -----------------------------
    pos_dissimilarity_prev: float = 0.0

    # -----------------------------
    # Lexical change (IMPROVED)
    # -----------------------------
    word_dissimilarity_prev: float = 0.0
    new_word_ratio: float = 0.0  # NEW

    # -----------------------------
    # Discourse (you already had)
    # -----------------------------
    causal_cohesion: float = 0.0
    intentional_cohesion: float = 0.0
    temporal_cohesion: float = 0.0

    # -----------------------------
    # Verb structure
    # -----------------------------
    verb_overlap_adjacent: float = 0.0
    verb_tense_repetition: float = 0.0
    verb_aspect_repetition: float = 0.0

    # -----------------------------
    # Diversity
    # -----------------------------
    type_token_ratio: float = 0.0
    lexical_diversity_all: float = 0.0
    lexical_diversity_verbs: float = 0.0

    # -----------------------------
    # Length / scaling (NEW)
    # -----------------------------
    sentence_length: float = 0.0
    sentence_length_log: float = 0.0
@dataclass
class SentenceFeatures:
    sentence_id: int
    text: str
    surface: SurfaceFeatures
    syntax: SyntaxFeatures
    lexical: LexicalFeatures
    cohesion: CohesionFeatures

@dataclass
class DocumentFeatures:
    all_content_lemmas: set = field(default_factory=set)
    all_noun_lemmas: set = field(default_factory=set)
    all_verb_lemmas: set = field(default_factory=set)
    all_stems: set = field(default_factory=set)
    all_argument_lemmas: set = field(default_factory=set)

    sentence_embeddings: np.ndarray = None 

    lexical_diversity_all: float = 0.0
    lexical_diversity_verbs: float = 0.0

    sentence_cache: list[dict] = field(default_factory=list)
    
    
@dataclass
class DocumentProfile:
    # Lexical features
    nouns: FeatureStats
    verbs: FeatureStats
    adjectives: FeatureStats
    adverbs: FeatureStats
    pronouns: FeatureStats
    first_person_pronouns: FeatureStats
    third_person_pronouns: FeatureStats

    avg_syllables_per_word: FeatureStats
    avg_word_frequency_log: FeatureStats
    avg_content_word_frequency_log: FeatureStats
    avg_age_of_acquisition: FeatureStats
    avg_concreteness: FeatureStats
    avg_imagery: FeatureStats
    avg_familiarity: FeatureStats
    avg_polysemy: FeatureStats
    negations: FeatureStats
    causal_verbs: FeatureStats
    intentional_actions: FeatureStats
    
    word_count: FeatureStats
    sentence_length: FeatureStats
    function_to_content_ratio: FeatureStats

    # connectives
    connectives_total: FeatureStats
    causal_connectives: FeatureStats
    temporal_connectives: FeatureStats
    logical_connectives: FeatureStats
    additive_connectives: FeatureStats
    adversative_connectives: FeatureStats
    
    dependency_depth: FeatureStats
    modifiers_per_np: FeatureStats
    words_before_main_verb: FeatureStats
    passive_constructions: FeatureStats
    
    content_overlap_adjacent: FeatureStats
    content_overlap_all: FeatureStats
    noun_overlap_adjacent: FeatureStats
    argument_overlap_adjacent: FeatureStats
    stem_overlap_all: FeatureStats

    lsa_overlap_adjacent: FeatureStats
    lsa_overlap_all: FeatureStats
    lsa_given_new: FeatureStats
    lsa_overlap_mean: FeatureStats
    lsa_overlap_std: FeatureStats
    lsa_overlap_max: FeatureStats
    lsa_overlap_min: FeatureStats
    lsa_novelty: FeatureStats
    lsa_shift: FeatureStats

    pos_dissimilarity_prev: FeatureStats
    word_dissimilarity_prev: FeatureStats
    new_word_ratio: FeatureStats

    verb_overlap_adjacent: FeatureStats
    embedding_norm: FeatureStats
    embedding_norm_diff: FeatureStats

    # verb_tense_repetition: FeatureStats
    # verb_aspect_repetition: FeatureStats

    type_token_ratio: FeatureStats
    lexical_diversity_all: FeatureStats
    lexical_diversity_verbs: FeatureStats
    sentence_length_cohesion: FeatureStats
    sentence_length_log: FeatureStats
