import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import re
from tqdm import tqdm

class LotteryScraper:
    def __init__(self, base_url="https://www.pilio.idv.tw/lto539/list.asp"):
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.total_pages = 241  # Total number of available pages

    def clean_numbers(self, text):
        """Clean and extract numbers from text"""
        # Remove Chinese characters and other non-numeric characters
        numbers = re.findall(r'\d+', text)
        return [int(num) for num in numbers if num.isdigit()]

    def scrape_page(self, page_number):
        """Scrape a single page of lottery results"""
        url = f"{self.base_url}?indexpage={page_number}&orderby=new"
        max_retries = 3
        retry_delay = 2

        for retry in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers)
                response.encoding = 'utf-8'  # Set correct encoding for Chinese characters
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find the table containing lottery data
                data_rows = []
                
                # Look for rows containing date and numbers
                for row in soup.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) == 2:
                        date_text = cells[0].get_text(strip=True)
                        numbers_text = cells[1].get_text(strip=True)
                        
                        # Extract date using regex
                        date_match = re.search(r'(\d{4}/\d{2}/\d{2})', date_text)
                        if date_match:
                            date = date_match.group(1)
                            # Clean and extract numbers
                            numbers = self.clean_numbers(numbers_text)
                            
                            if len(numbers) == 5:  # Ensure we have exactly 5 numbers
                                data_rows.append({
                                    'date': date,
                                    'number1': numbers[0],
                                    'number2': numbers[1],
                                    'number3': numbers[2],
                                    'number4': numbers[3],
                                    'number5': numbers[4]
                                })
                
                return pd.DataFrame(data_rows)
            
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error on page {page_number}, attempt {retry + 1}: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to scrape page {page_number} after {max_retries} attempts: {str(e)}")
                    return pd.DataFrame()

    def scrape_historical_data(self, start_page=1, end_page=None):
        """Scrape multiple pages of historical data"""
        if end_page is None:
            end_page = self.total_pages

        all_data = []
        total_pages = end_page - start_page + 1
        
        print(f"\nScraping {total_pages} pages of historical lottery data...")
        print("This might take a while. Large datasets will improve prediction accuracy.")
        
        # Create progress bar
        with tqdm(total=total_pages, desc="Scraping Progress") as pbar:
            for page in range(start_page, end_page + 1):
                page_data = self.scrape_page(page)
                if not page_data.empty:
                    all_data.append(page_data)
                time.sleep(0.5)  # Reduced delay but still being respectful
                pbar.update(1)
                
                # Save intermediate results every 50 pages
                if page % 50 == 0:
                    self.save_intermediate_results(all_data, page)

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data['date'] = pd.to_datetime(combined_data['date'])
            return combined_data.sort_values('date', ascending=False)
        
        return pd.DataFrame()

    def save_intermediate_results(self, data_list, current_page):
        """Save intermediate results to prevent data loss"""
        if data_list:
            intermediate_data = pd.concat(data_list, ignore_index=True)
            intermediate_data['date'] = pd.to_datetime(intermediate_data['date'])
            intermediate_data = intermediate_data.sort_values('date', ascending=False)
            filename = f'lottery_history_intermediate_p{current_page}.csv'
            intermediate_data.to_csv(filename, index=False)
            print(f"\nSaved intermediate results up to page {current_page}")

    def save_to_csv(self, data, filename='lottery_history.csv'):
        """Save the scraped data to a CSV file"""
        if not data.empty:
            data.to_csv(filename, index=False)
            print(f"\nFinal dataset saved to {filename}")
            print(f"Total records: {len(data)}")
            print(f"Date range: {data['date'].min()} to {data['date'].max()}")
        else:
            print("No data to save")

if __name__ == "__main__":
    scraper = LotteryScraper()
    # Scrape all available historical data
    historical_data = scraper.scrape_historical_data()
    if not historical_data.empty:
        scraper.save_to_csv(historical_data)
        print(f"Successfully scraped {len(historical_data)} lottery results") 