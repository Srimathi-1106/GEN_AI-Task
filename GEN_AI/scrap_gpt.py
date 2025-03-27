import time
import re
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import openai
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# Initialize OpenAI API (or your preferred LLM API)
# openai.api_key =  # Replace with your OpenAI key or another LLM provider's key


# Initialize the Selenium WebDriver
options = Options()
options.headless = True  # Run in headless mode (no browser UI)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Function to extract text content from a page
def get_page_content(url):
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

# Function to clean the scraped text (remove URLs, emails, ads, etc.)
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '', text)
    # Remove advertisement-related words
    text = re.sub(r'\b(advertisement|promo|sponsored)\b', '', text, flags=re.I)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to extract links to other pages
def get_links_from_page(url):
    driver.get(url)
    
    # Wait for the page to load completely
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

# Generate a prompt for LLM to extract specific information
def create_prompt(text_data):
    prompt = f"""
    Based on the following content from the company website, answer the following questions:

    1. What is the company's mission statement or core values?
    2. What products or services does the company offer?
    3. When was the company founded, and who were the founders?
    4. Where is the company's headquarters located?
    5. Who are the key executives or leadership team members?
    6. Has the company received any notable awards or recognitions?

    Content: 
    {text_data}
    """
    return prompt

# Function to get LLM response
def get_llm_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        prompt=prompt,
        max_tokens=100 
    )

# Extract and print the response
    print(response['choices'][0]['message']['content'])

# Function to scrape all pages starting from the base URL
def scrape_all_pages(base_url, max_pages=100):
    visited = set()  # Set to track visited pages
    to_visit = set([base_url])  # Set to keep track of pages to visit
    scraped_data = []

    count = 0
    while to_visit:
        if count >= max_pages:  # Limit to scraping 'max_pages' pages
            break
        
        url = to_visit.pop()  # Get the next URL to visit
        if url not in visited:
            visited.add(url)  # Mark this URL as visited
            count += 1
            print(f"Scraping {count}: {url}")
            
            # Get the content of the page
            raw_content = get_page_content(url)
            cleaned_content = clean_text(raw_content)
            print(f"Content from {url}: {cleaned_content[:200]}...")  # Print first 200 characters of the content
            
            # Create the LLM prompt
            prompt = create_prompt(cleaned_content)
            
            # Get LLM response
            llm_response = get_llm_response(prompt)
            print(f"LLM Response: {llm_response}")
            
            # Store the extracted details for this page
            extracted_data = {
                'URL': url,
                'Mission': llm_response.split("\n")[0],  # You can adjust based on LLM's format
                'Products/Services': llm_response.split("\n")[1],
                'Founded': llm_response.split("\n")[2],
                'Headquarters': llm_response.split("\n")[3],
                'Key Executives': llm_response.split("\n")[4],
                'Awards': llm_response.split("\n")[5]
            }
            scraped_data.append(extracted_data)

            # Get the links to other pages on the same site
            links = get_links_from_page(url)
            to_visit.update(links)  # Add new links to the to_visit set
            
            # Sleep to avoid making too many requests in a short time
            time.sleep(1)

    print("Scraping completed.")
    return scraped_data

# Save the extracted data into a CSV file
def save_to_csv(data, filename="company_details.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['URL', 'Mission', 'Products/Services', 'Founded', 'Headquarters', 'Key Executives', 'Awards'])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

# Main function to start scraping from a list of URLs
def main():
    base_url = 'https://www.snap.com'  # Replace with the desired URL
    scraped_data = scrape_all_pages(base_url)
    save_to_csv(scraped_data)

    # Quit the driver after the scraping is done
    driver.quit()

# Run the script
if __name__ == "__main__":
    main()
