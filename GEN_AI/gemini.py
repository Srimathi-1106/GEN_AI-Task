import requests
import time
import csv
from bs4 import BeautifulSoup
import google.generativeai as genai
 
 
# genai.configure(api_key=)
# model = genai.GenerativeModel("gemini-2.0-pro-exp")
 
 
 
questions = [
    "What is the company's mission statement or core values?",
    "What products or services does the company offer?",
    "When was the company founded, and who were the founders?",
    "Where is the company's headquarters located?",
    "Who are the key executives or leadership team members?",
    "Has the company received any notable awards or recognitions?"
]
 
 
def get_gemini_answers(text):
    """Ask Gemini to extract information from the scraped text."""
   
    # ✅ Ensure text is within a reasonable length (Gemini has token limits)
    text = text[:5000]  # Truncate text if too long
 
    prompt = f"Extract the following details from the text:\n\n{text}\n\n"
    for q in questions:
        prompt += f"- {q}\n"
 
    try:
        response = model.generate_content(prompt)  # ✅ Generate response
        return response.text if response and hasattr(response, "text") else "No response from Gemini"
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return "API Error"
 
 
main_url="https://www.tcs.com/who-we-are"
 
header={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0"}
 
response=requests.get(main_url,headers=header)
 
soup=BeautifulSoup(response.text,"html.parser")
page_text = soup.get_text(separator="\n", strip=True)

print(page_text)
 
# 🔹 Get extracted answers using Gemini
extracted_info = get_gemini_answers(page_text)
 
print("🔎 Extracted Information:")
print(extracted_info)