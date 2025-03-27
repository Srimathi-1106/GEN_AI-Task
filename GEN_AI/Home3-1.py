from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import re
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urljoin
import google.generativeai as genai

# Function to check if the URL is valid
def is_valid_url(url):
    return re.match(r'^(https?://)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(/[\w.-]*)*$', url) is not None

# Initialize Selenium WebDriver
options = Options()
options.headless = True
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Function to extract text content from a page
def get_page_content(url):
    try:
        if is_valid_url(url):
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Remove unnecessary tags
            for script in soup(["script", "style"]):
                script.decompose()

            return soup.get_text(separator=' ', strip=True)
        else:
            print(f"Skipping invalid URL: {url}")
            return ""
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return ""

# Function to extract links from header and footer
def get_header_footer_links(url):
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    header = soup.find('header')
    footer = soup.find('footer')

    links = set()
    for section in (header, footer):
        if section:
            for link in section.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                links.add(full_url)

    return links

# Function to scrape all pages
def scrape_all_pages(base_url):
    visited = set()
    to_visit = {base_url}
    full_content = []
    count = 0
    
    while to_visit and count < 10:
        url = to_visit.pop()
        if url not in visited:
            visited.add(url)
            count += 1
            print(f"Scraping {count}: {url}")

            content = get_page_content(url)
            full_content.append(content)
            
            # Get new links
            links = get_header_footer_links(url)
            to_visit.update(links)
            time.sleep(1)

    return "\n\n".join(full_content)

# Initialize Gemini API
# genai.configure(api_key=)
# model = genai.GenerativeModel("gemini-2.0-pro-exp")

# Questions for information extraction
questions = [
    "What is the company's mission statement or core values?",
    "What products or services does the company offer?",
    "When was the company founded, and who were the founders?",
    "Where is the company's headquarters located?",
    "Who are the key executives or leadership team members?",
    "Has the company received any notable awards or recognitions?"
]

# Function to extract information using Gemini model
def get_gemini_answers(text, question):
    prompt = f"Give answer for the given question from the text as short and crisp:\n\n{text}\n\n- {question}"
    try:
        response = model.generate_content(prompt)
        return response.text if response and hasattr(response, "text") else "No response from Gemini"
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "API Error"

# List of URLs to scrape
urls = [
    "https://www.snap.com",
    "https://www.dropbox.com",
    "https://www.tesla.com",
    "https://www.spacex.com",
    "https://robinhood.com",
    "https://stripe.com",
    "https://squareup.com",
    "https://www.shopify.com",
    "https://www.zara.com",
    "https://hmgroup.com"
]

# Process each URL
for c, url in enumerate(urls, start=1):
    print(f"\nProcessing {url}...\n")
    
    # Scrape content
    all_scraped_content = scrape_all_pages(url)

    # Save to text file
    with open(f"scraped_content_{c}.txt", "w", encoding="utf-8") as file:
        file.write(all_scraped_content)

    # Extract information using Gemini
    results = [{"Question": q, "Answer": get_gemini_answers(all_scraped_content, q)} for q in questions]

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"questions_and_answers_{c}.csv", index=False, encoding="utf-8")

# Close WebDriver
driver.quit()
