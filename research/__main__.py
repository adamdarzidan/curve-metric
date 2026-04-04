import scraper as s
from ..components.data_module import HospitalStruct, ScoreStruct


def main():
    hospitals: list[HospitalStruct] = s.scrape()
    
if __name__ == "__main__":
    main()
    
