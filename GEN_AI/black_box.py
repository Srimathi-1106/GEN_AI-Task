import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

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

# Function to initialize the Selenium WebDriver
def init_driver():
    options = Options()
    options.headless = True  # Run in headless mode (no browser UI)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)options=chrome_options)
    return driver

# Function to scrape navigation links from header and footer
def scrape_navigation_links(driver, url):
    try:
        driver.get(url)
        time.sleep(3)  # Wait for the page to load

        # Get the page source and parse it with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract header links
        header_links = []
        header = soup.find('header')
        if header:
            for link in header.find_all('a', href=True):
                header_links.append(link['href'])

        # Extract footer links
        footer_links = []
        footer = soup.find('footer')
        if footer:
            for link in footer.find_all('a', href=True):
                footer_links.append(link['href'])

        return {
            "url": url,
            "header_links": header_links,
            "footer_links": footer_links
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {"url": url, "header_links": [], "footer_links": []}

# Main function to orchestrate the scraping
def main():
    driver = init_driver()
    results = []
    
    for url in urls:
        navigation_links = scrape_navigation_links(driver, url)
        results.append(navigation_links)
    
    driver.quit()  # Close the browser

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('navigation_links.csv', index=False)

if __name__ == "__main__":
    main()