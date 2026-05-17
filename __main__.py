from components.metric import Metric

import util
from config import PathConfig, UIConfig
import json


def train_command(model: Metric):
    # Get valid index for csv file
    idx = util.get_valid_index(PathConfig.TRAIN_FILES)
    
    # Find size of training set
    size = int(s) if (s := input("Enter max number of training sets you want to train for. Enter none for all: ")).isdigit() else -1
    
    # Construct CSV path
    path = PathConfig.TRAIN_PATH + PathConfig.TRAIN_FILES[idx]
    
    # Begin training, once finished, returns
    model.train(path, size if size > 0 else 10000)
    
def validate_command(model: Metric):
    # Get valid index for csv file
    idx = util.get_valid_index(PathConfig.TRAIN_FILES)
    
    # Find size of training set
    size = int(s) if (s := input("Enter max number of testing sets you want to test for. Enter none for all: ")).isdigit() else -1
    
    # Construct CSV path
    path = PathConfig.TRAIN_PATH + PathConfig.TRAIN_FILES[idx]
    
    # Begin training, once finished, returns
    return model.validate(path, size if size > 0 else 10000)
    
        
def load_command(model: Metric):
    # Get valid index for model
    idx = util.get_valid_index(PathConfig.LOAD_FILES)
    
    # Configure path
    path = PathConfig.LOAD_PATH + PathConfig.LOAD_FILES[idx]
    
    # Load model and return
    model.load_model(path)
        
def test_command(model: Metric) -> bool:
    # Exit if no model is loaded
    util.handle_error("ERROR: Model not loaded", False) if not model.loaded else None
    
    idx = util.get_valid_index(PathConfig.TEST_FILES)
    
    with open(PathConfig.TEST_PATH + PathConfig.TEST_FILES[idx]) as f:
        text = f.read()
    
    print(f"Model score: {model.score(text, print_shap=True)}")
        

  
def main():
    print("Loading Model Object...")
    model = Metric()
    util.space()
    print("Model object loaded!")
    util.space()
    quit = False
    while(not quit):
        util.print_ui(model)
        cmd = util.get_valid_input("", UIConfig.VALID_COMMANDS)
        
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
            case "validate":
                util.space()
                print(validate_command(model))
        
    util.space()
    print("QUIT PROCESS")
    util.space()
    
if __name__ == "__main__":
    main()