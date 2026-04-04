from components.metric import Metric

import util
from config import Config
import json


def train_command(model: Metric):
    # Get valid index for csv file
    idx = util.get_valid_index(Config.train_files)
    
    # Find size of training set
    size = int(s) if (s := input("Enter max number of training sets you want to train for. Enter none for all: ")).isdigit() else -1
    
    # Construct CSV path
    path = Config.train_path + Config.train_files[idx]
    
    # Begin training, once finished, returns
    model.train(path, size if size <= 0 else 500)
    
        
def load_command(model: Metric):
    # Get valid index for model
    idx = util.get_valid_index(Config.load_files)
    
    # Configure path
    path = Config.load_path + Config.load_files[idx]
    
    # Load model and return
    model.load_model(path)
        
def test_command(model: Metric) -> bool:
    # Exit if no model is loaded
    util.handle_error("ERROR: Model not loaded", False) if not model.loaded else None
    
    while(True):
        # Grab index of test file
        idx = util.get_valid_index(Config.test_files)
        
        # Fill in rest of logic later
        # -------------
        return False

def perform_extraction(model: Metric):
    path = "hospitals.json"
    with open(path) as file:
        data = json.load(file)
        hospitals = data["hospitals"]

    

  
def main():
    print("Loading Model Object...")
    model = Metric([0])
    util.space()
    print("Model object loaded!")
    util.space()
    quit = False
    while(not quit):
        util.print_ui(model)
        cmd = util.get_valid_input("", Config.VALID_COMMANDS)
        
        match cmd.lower():
            case "quit":
                quit = True
            case "train":
                util.space()
                if(train_command(model)):
                    quit = True
            case "load":
                util.space()
                if(load_command(model)):
                    quit = True
            case "test":
                util.space()
                if(test_command(model)):
                    quit = True
            case "extract":
                util.space()
                if(perform_extraction(model)):
                    quit = True
                    
    util.space()
    print("QUIT PROCESS")
    util.space()
    
if __name__ == "__main__":
    main()