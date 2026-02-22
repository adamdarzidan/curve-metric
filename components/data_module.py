from dataclasses import dataclass

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
    function_to_content_ratio: float = 0.0

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
    modifiers_per_np: float = 0.0
    words_before_main_verb: int = 0
    passive_constructions: int = 0
    syntactic_similarity_prev: float = 0.0  # optional for later
    
@dataclass
class CohesionFeatures:
    content_overlap_adjacent: float = 0.0
    content_overlap_all: float = 0.0
    noun_overlap_adjacent: float = 0.0
    argument_overlap_adjacent: float = 0.0
    stem_overlap_all: float = 0.0

    lsa_overlap_adjacent: float = 0.0
    lsa_overlap_all: float = 0.0
    lsa_given_new: float = 0.0
    lsa_verb_overlap_adjacent: float = 0.0

    pos_dissimilarity_prev: float = 0.0
    word_dissimilarity_prev: float = 0.0

    causal_cohesion: float = 0.0
    intentional_cohesion: float = 0.0
    temporal_cohesion: float = 0.0

    verb_overlap_adjacent: float = 0.0
    verb_tense_repetition: float = 0.0
    verb_aspect_repetition: float = 0.0

    type_token_ratio: float = 0.0
    lexical_diversity_all: float = 0.0
    lexical_diversity_verbs: float = 0.0
    
@dataclass
class SentenceFeatures:
    sentence_id: int
    text: str
    surface: SurfaceFeatures
    syntax: SyntaxFeatures
    lexical: LexicalFeatures
    cohesion: CohesionFeatures