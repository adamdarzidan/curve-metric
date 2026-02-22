from .linguistic_processor import LinguisticProcessor
from .feature_profiler import FeatureProfiler
from .data_module import SentenceFeatures

class Metric:
    def __init__(self, text: str, profile: list[int]):
        self.text = text
        self.profile = profile
        
    def run(self):
        lp = LinguisticProcessor()
        fp = FeatureProfiler(lp)
        sentence_feature_list: list[SentenceFeatures] = fp.extract(self.text)
        
        