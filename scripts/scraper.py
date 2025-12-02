import pandas as pd
from google_play_scraper import app, reviews, Sort
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GooglePlayReviewScraper:
    def __init__(self):
        self.apps = {
            'cbe': {
                'app_id': 'com.cbe.mobile.banking.cbe',
                'name': 'Commercial Bank of Ethiopia (CBE)'
            },
            'boa': {
                'app_id': 'com.app.abyssinia',
                'name': 'Bank of Abyssinia (BOA)'
            },
            'dashen': {
                'app_id': 'com.dashenmobile',
                'name': 'Dashen Bank'
            }
        }
    
    def scrape_reviews(self, app_id, app_name, count=500, lang='en', country='et'):
        """Scrape reviews from Google Play Store"""
        try:
            logger.info(f"Scraping reviews for {app_name}...")
            
            # Get app info
            app_info = app(app_id, lang=lang, country=country)
            logger.info(f"App: {app_info['title']}, Rating: {app_info['score']}")
            
            # Scrape reviews
            all_reviews = []
            continuation_token = None
            
            for batch in range(0, count, 200):
                batch_size = min(200, count - batch)
                
                result, continuation_token = reviews(
                    app_id,
                    lang=lang,
                    country=country,
                    sort=Sort.NEWEST,
                    count=batch_size,
                    continuation_token=continuation_token
                )
                
                all_reviews.extend(result)
                
                if not continuation_token:
                    break
                
                time.sleep(2)  # Be respectful with requests
            
            # Process reviews
            processed_reviews = []
            for review in all_reviews:
                processed_reviews.append({
                    'review_id': review['reviewId'],
                    'review_text': review['content'],
                    'rating': review['score'],
                    'date': review['at'].strftime('%Y-%m-%d'),
                    'app_name': app_name,
                    'thumbs_up': review['thumbsUpCount'],
                    'reviewer_name': review['userName'],
                    'reviewer_image': review['userImage'],
                    'reply_text': review.get('replyContent', ''),
                    'reply_date': review.get('repliedAt', ''),
                    'source': 'Google Play Store'
                })
            
            logger.info(f"Scraped {len(processed_reviews)} reviews for {app_name}")
            return processed_reviews
            
        except Exception as e:
            logger.error(f"Error scraping {app_name}: {str(e)}")
            return []
    
    def scrape_all_apps(self, reviews_per_app=400):
        """Scrape reviews for all apps"""
        all_data = []
        
        for app_key, app_info in self.apps.items():
            reviews_data = self.scrape_reviews(
                app_info['app_id'],
                app_info['name'],
                count=reviews_per_app
            )
            all_data.extend(reviews_data)
            
            # Add delay between apps
            time.sleep(5)
        
        return pd.DataFrame(all_data)
    
    def save_to_csv(self, df, filename='data/raw/google_play_reviews_raw.csv'):
        """Save scraped data to CSV"""
        df.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"Saved {len(df)} reviews to {filename}")
        return filename

if __name__ == "__main__":
    scraper = GooglePlayReviewScraper()
    
    # Scrape reviews
    reviews_df = scraper.scrape_all_apps(reviews_per_app=400)
    
    # Save raw data
    scraper.save_to_csv(reviews_df)
    
    # Display summary
    print("\n=== Scraping Summary ===")
    print(f"Total Reviews: {len(reviews_df)}")
    print("\nReviews per app:")
    print(reviews_df['app_name'].value_counts())
    print("\nRating distribution:")
    print(reviews_df['rating'].value_counts().sort_index())