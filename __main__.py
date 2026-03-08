import json  
import sys
import os
from components.metric import Metric

from termcolor import colored
from colorama import init

init()

TRAIN_FILE_PATH = "data/train/"
LOAD_FILE_PATH = "weights/"
TEST_FILE_PATH = "data/texts/"

TRAIN_FILENAMES = [f.lower() for f in os.listdir(TRAIN_FILE_PATH)]
LOAD_FILENAMES = [f.lower() for f in os.listdir(LOAD_FILE_PATH)]
TEST_FILENAMES = [f.lower() for f in os.listdir(TEST_FILE_PATH)]


VALID_COMMANDS = ["train", "quit", "load", "test", "back"]

BOLD = '\033[1m'
END = '\033[0m'

def format_title(title: str | None, total_length: int = 60) -> str:
    if title is None:
        return "-" * total_length
    remainder = total_length - len(title)
    return ("-" * int(remainder / 2)) + title + ("-" * int(remainder / 2))

def valid_command(cmd: str) -> bool:
    if(not(cmd.lower() in VALID_COMMANDS)):
        return False
    return True

def train_command(model: Metric) -> bool:
    while(True):
        print(format_title("Enter index for dataset"))
        print([f"{i+1}. {TRAIN_FILENAMES[i]}" for i in range(len(TRAIN_FILENAMES))])
        print(format_title(None))
        
        try:
            idx = int(input())
        except:
            return False
        
        while((idx) < 1 or ((idx) > len(TRAIN_FILENAMES))):
            try:
                int(input(f"Enter valid index from {TRAIN_FILENAMES}"))
            except:
                return False
            
        try:
            num_texts = int(input("Enter max number of training sets you want to train for. Enter none for all: "))
        except:
            num_texts = -1
        
        model.train(TRAIN_FILE_PATH + TRAIN_FILENAMES[idx - 1], num_texts if num_texts != -1 else 500)
        return False
        
def load_command(model: Metric) -> bool:
    while(True):
            print(format_title("Enter index for saved file"))
            print([f"{i+1}. {LOAD_FILENAMES[i]}" for i in range(len(LOAD_FILENAMES))])
            print(format_title(None))
            
            try:
                idx = int(input())
            except:
                return True
            
            while((idx) < 1 or ((idx) > len(LOAD_FILENAMES))):
                try:
                    idx = int(input(f"Enter valid index from {LOAD_FILENAMES}"))
                except:
                    return True
            
            model.load_model(LOAD_FILE_PATH + LOAD_FILENAMES[idx - 1])
            return False
        
def test_command(model: Metric) -> bool:
    if not model.loaded:
        print("\n ERROR: No model loaded")
        return False
    
    while(True):
        print(format_title("Enter index for testing file"))
        print([f"{i+1}. {TEST_FILENAMES[i]}" for i in range(len(TEST_FILENAMES))])
        print(format_title(None))
        
        try:
            idx = int(input())
        except:
            return True
        
        while((idx) < 1 or ((idx) > len(TEST_FILENAMES))):
            try:
                idx = int(input(f"Enter valid index from {TEST_FILENAMES}"))
            except:
                return True
        path = TEST_FILE_PATH + TEST_FILENAMES[idx - 1]
        
        with open(TEST_FILE_PATH + TEST_FILENAMES[idx - 1]) as f:
            text = f.read()
            
        print("\n\n\n", format_title(None))
        print("Text: ", text)
        print(f"\n{format_title(None)}\n{BOLD}Text score:{END} {colored(model.score(text), "red")}\n{format_title(None)}\n\n")
        return False
        
  
def main():
    print("Loading Model Object...")
    model = Metric([0])
    print("\n\nModel object loaded!\n")
    quit = False
    while(not quit):
        print("\n", format_title("METRIC SYSTEM"))
        print("CMDS:")
        print(colored("QUIT", "red"), " - Quit Program")
        print(colored("TRAIN", "green"), " - Train the model | params: csv_file")
        print(colored("TEST", "blue") ," - Test a model | params: text, model_file")
        print(colored("LOAD", "yellow")," - Load a pre-existing model | params: file_path")
        print(format_title(None))
        cmd = input()
        while(not valid_command(cmd)):
            cmd = input(f"Enter a valid command: {VALID_COMMANDS}\n")
            
        print("\n\n")
        match cmd.lower():
            case "quit":
                quit = True
            case "train":
                if(train_command(model)):
                    quit = True
            case "load":
                if(load_command(model)):
                    quit = True
            case "test":
                if(test_command(model)):
                    quit = True
   
    print("QUIT PROCESS")
    
if __name__ == "__main__":
    main()