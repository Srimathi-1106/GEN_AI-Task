from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


import requests
import time
import csv
from bs4 import BeautifulSoup
import google.generativeai as genai

# Initialize the Selenium WebDriver
options = Options()
options.headless = True  # Run in headless mode (no browser UI)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Function to extract text content from a page
def get_page_content(url):
    driver.get(url)  # Open the page using Selenium
    
    # Wait for the page to load completely (e.g., waiting for a specific element)
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

# Function to extract links to other pages
def get_links_from_page(url):
    driver.get(url)
    
    # Wait for the page to load completely (e.g., waiting for the body to load)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # Find all links (<a> tags) on the page
    links = set()
    anchors = driver.find_elements(By.TAG_NAME, 'a')  # Find all anchor tags
    for anchor in anchors:
        href = anchor.get_attribute('href')
        if href and "snap.com" in href:  # Filter only internal links
            links.add(href)
    
    return links

full_content = []  # Store content in a list to join later

# Function to scrape all pages starting from the base URL
def scrape_all_pages(base_url):
    visited = set()  # Set to track visited pages
    to_visit = set([base_url])  # Set to keep track of pages to visit
    count = 0
    
    while to_visit:
        if count >= 10:  # Limit to scraping 10 pages for now (you can adjust this)
            break
        
        url = to_visit.pop()  # Get the next URL to visit
        if url not in visited:
            visited.add(url)  # Mark this URL as visited
            count += 1
            print(f"Scraping {count}: {url}")
            
            # Get the content of the page
            content = get_page_content(url)
            print(f"Content from {url}: {content[:200]}...")  # Print first 200 characters of the content
            full_content.append(content)  # Append content to the list
            
            # Get the links to other pages on the same site
            links = get_links_from_page(url)
            to_visit.update(links)  # Add new links to the to_visit set
            
            # Sleep to avoid making too many requests in a short time
            time.sleep(1)

    print("Scraping completed.")

# Start scraping from the base URL
base_url = 'https://www.snap.com'
scrape_all_pages(base_url)

# Combine all the content into a single string
all_scraped_content = "\n\n".join(full_content)
print(all_scraped_content[:500])  # Print first 500 characters of all content

# Quit the driver after the scraping is done


 
# genai.configure(api_key="")
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
    
extracted_info = get_gemini_answers(all_scraped_content)

print(extracted_info)
driver.quit()