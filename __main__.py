import json  
import sys
from components.metric import Metric
  
def main(file_name: str):
    try:
        with open(file=file_name) as file:
            data = json.load(file)
            text = data["text"]
            profile = [0, 0.5, 1, 1.2]
            
            obj = Metric(text, profile)
            obj.run()
            
        
    except FileNotFoundError:
        print(f"Error: File name {file_name} was not found")
    except json.JSONDecodeError:
        print(f"Error: failed to decode file ({file_name})")
    
if __name__ == "__main__":
    if(len(sys.argv) <= 1):
        raise Exception("Error: no arguments provided. Must have index file argument")
    main(sys.argv[1])