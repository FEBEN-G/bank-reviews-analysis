import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.scraper import GooglePlayReviewScraper
from scripts.preprocessor import ReviewPreprocessor
import logging

def main():
    """Execute Task 1: Data Collection and Preprocessing"""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("TASK 1: DATA COLLECTION AND PREPROCESSING")
    print("=" * 60)
    
    # Step 1: Scrape reviews
    print("\n1. Scraping reviews from Google Play Store...")
    scraper = GooglePlayReviewScraper()
    reviews_df = scraper.scrape_all_apps(reviews_per_app=400)
    
    # Save raw data
    raw_file = scraper.save_to_csv(reviews_df)
    print(f"✓ Raw data saved to: {raw_file}")
    
    # Step 2: Preprocess data
    print("\n2. Preprocessing scraped data...")
    preprocessor = ReviewPreprocessor()
    processed_df = preprocessor.preprocess_dataframe(reviews_df)
    
    # Save processed data
    processed_file = preprocessor.save_processed_data(processed_df)
    print(f"✓ Processed data saved to: {processed_file}")
    
    # Step 3: Generate summary
    print("\n3. Generating summary...")
    summary = preprocessor.generate_summary(processed_df)
    
    print("\n" + "=" * 60)
    print("TASK 1 COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Total Reviews Collected: {summary['total_reviews']}")
    print(f"Target Achieved: {'✓' if summary['total_reviews'] >= 1200 else '✗'}")
    print(f"Missing Data: {summary['missing_data_percentage']:.2f}%")
    print(f"Quality Check: {'✓' if summary['missing_data_percentage'] < 5 else '✗'}")
    
    # Save summary to file
    import json
    with open('data/processed/task1_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to: data/processed/task1_summary.json")

if __name__ == "__main__":
    main()