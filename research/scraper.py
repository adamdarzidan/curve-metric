import requests
import pdfplumber
import pandas as pd
import textstat
import os
import time
import json
from config import Config
from config import GOOGLE_CLOUD_TOKEN
from config import SEARCH_ENGINE_ID
from io import BytesIO
import util

from ..components.data_module import HospitalStruct


def get_data(path: str) -> list[HospitalStruct]:
    try:
        with open(path) as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Hospital data file does not exist")
        return []

    return [
        HospitalStruct(h["name"], h["state"], h["type"], h["rank"])
        for h in data["hospitals"]
    ]


def get_hippa_documents(self, hospital_name):
    query = f"{hospital_name} notice of privacy practices filetype:pdf"
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_CLOUD_TOKEN,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": 3
    }
    response = requests.get(url, params=params)
    results = response.json().get("items", [])
    
    for item in results:
        link: str = item.get("link", "")
        if link.endswith(".pdf"):
            return link
    return None    


def extract_text_from_pdf(pdf_url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(pdf_url, headers=headers, timeout=15)
        
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
        return text.strip()
    except Exception as e:
        print(f"Failed: {pdf_url} - {e}")
        return None
    



def scrape(data_path: str = Config.HOSPITAL_DATA_PATH, display_failed: bool = True):
    hospitals: list[HospitalStruct] = get_data(data_path)
    failed_list: list[str] = []
    
    for hospital in hospitals:
        
        pdf_link = get_hippa_documents(hospital.name)
        # Check if extraction was succesful, if not remove from list
        if pdf_link == None:
            failed_list.append(hospital.name)
            hospitals.remove(hospital)
            continue
        
        hospital.pdf_link = pdf_link
        text = extract_text_from_pdf(pdf_link)
        # Check if extraction was succesful, if not remove from list
        if text == None:
            failed_list.append(hospital.name)
            hospitals.remove(hospital)
            continue
        
    if display_failed:
            util.space(1)
            print("Failed extractions:")
            for failed in failed_list:
                print(failed)
            util.space(1)
    
         
    return hospitals
    
