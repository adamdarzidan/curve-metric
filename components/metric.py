class Metric:
    def __init__(self, text: str, profile: list[int]):
        self.text = text
        self.profile = profile
        
    def run(self):
        print(self.text)