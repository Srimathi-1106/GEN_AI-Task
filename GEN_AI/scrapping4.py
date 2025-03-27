import csv
import time
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urljoin

# Function to check if the URL is valid
def is_valid_url(url):
    return re.match(r'^(https?://)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(/[\w.-]*)*$', url) is not None

# Initialize the Selenium WebDriver
options = Options()
options.headless = True  # Run in headless mode (no browser UI)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Function to extract text content from a page
def get_page_content(url):
    try:
        if is_valid_url(url):
            driver.get(url)  # Open the page using Selenium
            
            # Wait for the page to load completely (e.g., waiting for the body to load)
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
        else:
            print(f"Skipping invalid URL: {url}")
            return ""  # Return empty content for invalid URLs
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return ""  # Return empty content in case of error

# Function to extract links from header and footer sections
def get_header_footer_links(url):
    driver.get(url)
    
    # Wait for the page to load completely
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # Parse the page with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Find the header and footer sections
    header = soup.find('header')  # Look for header tag
    footer = soup.find('footer')  # Look for footer tag

    # Extract all links in header and footer
    links = set()

    if header:
        header_links = header.find_all('a', href=True)  # Find all links in header
        for link in header_links:
            href = link['href']
            if href:
                # Join relative URLs with the base URL
                full_url = urljoin(url, href)
                links.add(full_url)

    if footer:
        footer_links = footer.find_all('a', href=True)  # Find all links in footer
        for link in footer_links:
            href = link['href']
            if href:
                # Join relative URLs with the base URL
                full_url = urljoin(url, href)
                links.add(full_url)

    return links

# List to store scraped contents
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
            full_content.append((url, content))  # Append URL and content to the list
            
            # Get the links to other pages in the header and footer
            links = get_header_footer_links(url)
            to_visit.update(links)  # Add new links to the to_visit set
            
            # Sleep to avoid making too many requests in a short time
            time.sleep(1)

    print("Scraping completed.")

# Start scraping from the base URL
base_url = 'https://www.snap.com'
scrape_all_pages(base_url)

# Define questions to extract from the scraped content
questions = [
    "What is the company's mission statement or core values?",
    "What products or services does the company offer?",
    "When was the company founded, and who were the founders?",
    "Where is the company's headquarters located?",
    "Who are the key executives or leadership team members?",
    "Has the company received any notable awards or recognitions?"
]

# Initialize Gemini API (make sure you have valid API key)
# genai.configure(api_key="your_api_key_here")
# model = genai.GenerativeModel("gemini-2.0-pro-exp")

# Function to extract information using Gemini model
def get_gemini_answers(question, text):
    """Ask Gemini to extract information from the scraped text."""
   
    # Ensure text is within a reasonable length (Gemini has token limits)
    text = text[:5000]  # Truncate text if too long
    
    # Construct the prompt for the specific question
    prompt = f"Extract the following detail from the text:\n\n{text}\n\n{question}\n"
    
    try:
        response = model.generate_content(prompt)  # Generate response
        return response.text if response and hasattr(response, "text") else "No response from Gemini"
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "API Error"

# Prepare the data for CSV
csv_columns = [
    "URL",
    "Mission Statement or Core Values",
    "Products/Services",
    "Founding Date and Founders",
    "Headquarters Location",
    "Key Executives",
    "Notable Awards or Recognitions"
]

csv_data = []

# Iterate over the scraped content and get the answers for each question
for url, content in full_content:
    row = {"URL": url}
    
    # For each question, get the response from the AI and store in the row
    for question in questions:
        answer = get_gemini_answers(question, content)
        row[question] = answer
    
    csv_data.append(row)

# Write the data to a CSV file
with open("scraped_info.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(csv_data)

# Close the Selenium driver
driver.quit()
