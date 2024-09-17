import os
import time
import requests
from typing import Dict, List, Tuple
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

def initialize_driver() -> webdriver.Chrome:
    """Initialize and return a Chrome WebDriver with custom options."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    return webdriver.Chrome(options=chrome_options)

def download_image(url: str, filename: str) -> bool:
    """Download an image from the given URL and save it with the specified filename."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return False

def extract_image_info(element: webdriver.remote.webelement.WebElement) -> Tuple[str, str, str]:
    """Extract image URL, title, and cadre from the given element."""
    img_url = element.get_attribute('href')
    title_attr = element.get_attribute('title')
    title_parts = title_attr.split('<br>')
    
    title = title_parts[0].strip()
    cadre = title_parts[2].replace('Cadre:', '').strip() if len(title_parts) >= 3 else "Unknown"
    
    return img_url, title, cadre

def clean_filename(filename: str) -> str:
    """Clean the filename by removing or replacing invalid characters."""
    return ''.join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in filename)

def main():
    """Main function to orchestrate the scraping and downloading process."""
    base_url = "https://www.lbsnaa.gov.in/course/participants_profile.php?cid=230&page={}"
    download_path = os.path.join(os.getcwd(), "99_FC_downloaded_images")
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    driver = initialize_driver()

    try:
        for page in range(1, 56):  # 55 pages in total
            url = base_url.format(page)

            driver.get(url)
            
            # Wait for the images to load
            WebDriverWait(driver, 30).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.fancybox"))
            )

            # Find all image elements
            image_elements = driver.find_elements(By.CSS_SELECTOR, "a.fancybox")
            
            total_images = len(image_elements)
            print(f"Found {total_images} images.")
            print(f"\nPage {page}:")
            #print(f"Found {len(elements)} matching elements.")

            for index, element in enumerate(image_elements, start=1):
                img_url, title, cadre = extract_image_info(element)
                
                filename = f"{clean_filename(title)}_{clean_filename(cadre)}_2024.jpg"
                full_path = os.path.join(download_path, filename)
                
                if download_image(img_url, full_path):
                    print(f"Downloaded {index}/{total_images}: {filename}")
                else:
                    print(f"Failed to download {index}/{total_images}: {filename}")

                time.sleep(1)  # Short delay between downloads
            time.sleep(2) # # Add a short delay between pages to avoid overwhelming the server

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        driver.quit()

    print(f"\nDownload process completed. Images are saved in: {download_path}")

if __name__ == "__main__":
    main()

