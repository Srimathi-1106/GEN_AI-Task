import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import google.generativeai as genai
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configure the Gemini API
# genai.configure(api_key="AIzaSyAVvDeZcm78AIVaGM3Q4to5f5RV7zlGOoc")  # Replace with your actual API key
# model = genai.GenerativeModel("gemini-2.0-pro-exp")

# Questions to ask the Gemini model
questions = [
    "What is the company's mission statement or core values?",
    "What products or services does the company offer?",
    "When was the company founded, and who were the founders?",
    "Where is the company's headquarters located?",
    "Who are the key executives or leadership team members?",
    "Has the company received any notable awards or recognitions?"
]

# Initialize the Selenium WebDriver
options = Options()
options.headless = True  # Run in headless mode (no browser UI)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Function to extract text content from a page
def get_page_content(url):
    driver.get(url)  # Open the page using Selenium
    
    # Wait for the page to load completely
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # Get the page source and parse with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Remove unnecessary tags like script or style
    for script in soup(["script", "style"]):
        script.decompose()

    # Extract the text content
    text = soup.get_text(separator=' ', strip=True)
    return text

# Function to extract links from header and footer
def get_links_from_header_footer(url):
    driver.get(url)
    
    # Wait for the page to load completely
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # Find header and footer elements
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    links = set()

    # Extract links from the header
    header = soup.find('header')
    if header:
        for anchor in header.find_all('a', href=True):
            href = anchor['href']
            if href.startswith('/'):
                href = url + href  # Convert relative links to absolute
            links.add(href)

    # Extract links from the footer
    footer = soup.find('footer')
    if footer:
        for anchor in footer.find_all('a', href=True):
            href = anchor['href']
            if href.startswith('/'):
                href = url + href  # Convert relative links to absolute
            links.add(href)

    return links

# Function to get answers from the Gemini model
def get_gemini_answers(text):
    """Ask Gemini to extract information from the scraped text."""
    text = text[:5000]  # Truncate text if too long

    prompt = f"Extract the following details from the text:\n\n{text}\n\n"
    for q in questions:
        prompt += f"- {q}\n"

    try:
        response = model.generate_content(prompt)  # Generate response
        return response.text if response and hasattr(response, "text") else "No response from Gemini"
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return "API Error"

# Function to scrape all pages starting from the base URL
def scrape_all_pages(base_url):
    visited = set()  # Set to track visited pages
    to_visit = set([base_url])  # Set to keep track of pages to visit
    results = []  # Store results for each website
    
    while to_visit:
        url = to_visit.pop()  # Get the next URL to visit
        if url not in visited:
            visited.add(url)  # Mark this URL as visited
            print(f"Scraping: {url}")
            
            # Get the content of the page
            content = get_page_content(url)
            print(f"Content from {url}: {content[:200]}...")  # Print first 200 characters of the content
            
            # Get answers from the Gemini model
            answers = get_gemini_answers(content)
            print(f"Answers from Gemini for Website {url} : \n {answers}")