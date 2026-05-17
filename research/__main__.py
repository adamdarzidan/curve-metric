
import pandas as pd
from .analyzer import Analyzer
from .text_processor import process_text
from .scraper import process_hospital_documents
import sys

from .config import HOSPITAL_DATA_PATH, DATAFRAME_PATH



def main():
    # Load the dataframe
    
    df = pd.DataFrame()
    try:
        df = pd.read_csv(f"{DATAFRAME_PATH}{input("Enter valid csv dataframe: enter -1 to process text:\n")}")
    except:
        df = process_text(HOSPITAL_DATA_PATH, save=True)
    # Create analyzer object
    a: Analyzer = Analyzer(df)
    
    # Plots generated
    a.generate_full_report()
if __name__ == "__main__":
    main()
    
