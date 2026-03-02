from .data_module import SentenceFeatures, DocumentProfile


class DocumentProfilier:
    def __init__(self, sf: list[SentenceFeatures]):
        self.sentence_features = sf
        
    def extract(self) -> DocumentProfile:
        length = len(self.sentence_features)
        profile = DocumentProfile()
        
        profile.additive_connectives = []
        
        return profile
    