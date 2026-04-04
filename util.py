from typing import TYPE_CHECKING
from termcolor import colored
from colorama import init

import sys

if TYPE_CHECKING:
    from components.metric import Metric


@staticmethod
def avg(n: float, m: float ):
    return n / m if m != 0 else 0   

@staticmethod
def get_valid_input(display, reqs) -> any:
    user_input = input(display)
    while(not(user_input.lower().strip() in reqs)):
        user_input = input(display)
    
    return user_input

def get_valid_index(options: list):
    space()
    format_title("Enter index to select")
    print([f"{i+1}. {options[i]}" for i in range(len(options))])
    
    s = input()

    while not (s.isdigit() and 1 <= int(s) <= len(options)):
        s = input()

    idx = int(s)
    return idx - 1

@staticmethod
def space(n = 2):
    for _ in range(n):
        print("\n")
  
  
@staticmethod  
def format_title(title: str | None, total_length: int = 60) -> str:
    if title is None:
        return "-" * total_length
    remainder = total_length - len(title)
    return ("-" * int(remainder / 2)) + title + ("-" * int(remainder / 2))

   
@staticmethod 
def print_ui(model: "Metric" ):    
    init()
    print("\n", format_title("METRIC SYSTEM"))
    print(f"LOADED MODE: {model.model_filepath if model.loaded else "NONE"} \nCMDS:")
    print(colored("QUIT", "red"), " - Quit Program")
    print(colored("TRAIN", "green"), " - Train the model | params: csv_file")
    print(colored("TEST", "blue") ," - Test a model | params: text, model_file")
    print(colored("LOAD", "yellow")," - Load a pre-existing model | params: file_path")
    print(format_title(None))
    

@staticmethod
def handle_error(message, exit = True):
    init()
    print(colored(message, "red"))
    if exit:
        sys.exit(1)
    return False