# Bank Reviews Analysis - Ethiopian Banking Apps

A comprehensive data analysis project for analyzing customer reviews of Ethiopian bank mobile applications from the Google Play Store.

## 📋 Project Overview

This project analyzes customer satisfaction with mobile banking apps by collecting and processing user reviews from the Google Play Store for three Ethiopian banks:

1. **Commercial Bank of Ethiopia (CBE)**
2. **Bank of Abyssinia (BOA)**
3. **Dashen Bank**

The project follows a 4-task structure as per assignment requirements, covering data collection, sentiment analysis, database implementation, and insights generation.

## 🎯 Business Objective

Omega Consultancy is supporting banks to improve their mobile apps to enhance customer retention and satisfaction. This analysis helps identify:

- Customer sentiment trends
- Key satisfaction drivers and pain points
- Thematic patterns in user feedback
- Actionable recommendations for improvement

## 📊 Project Structure

### Task 1: Data Collection and Preprocessing
- **Scripts**: `scripts/scraper.py`, `scripts/preprocessor.py`, `scripts/task1_main.py`
- **Output**: `data/processed/reviews_processed.csv` (1,200+ reviews)
- **Key Features**:
  - Web scraping using `google-play-scraper`
  - Data cleaning and preprocessing
  - Removal of duplicates and missing values
  - Date normalization and formatting

### Task 2: Sentiment and Thematic Analysis
- **Scripts**: `scripts/sentiment_analyzer.py`, `scripts/thematic_analyzer.py`, `scripts/task2_main.py`
- **Output**: `data/processed/reviews_with_themes.csv`
- **Key Features**:
  - Ensemble sentiment analysis (DistilBERT + VADER + TextBlob)
  - Thematic analysis with 8 predefined banking themes
  - Keyword extraction using TF-IDF
  - 100% sentiment coverage achieved
  - 8+ themes identified per bank

### Task 3: PostgreSQL Database Implementation
- **Scripts**: `scripts/task3_main.py`, `database/schema.sql`
- **Output**: `reports/task3_database_report.json`
- **Key Features**:
  - PostgreSQL database schema for bank reviews
  - Two main tables: `banks` and `reviews`
  - Support for 400+ reviews (actual: 1,200+)
  - Sentiment data integration
  - Complete Python database interface

### Task 4: Insights and Recommendations
- **Scripts**: `scripts/insights_analyzer.py`, `scripts/visualization_generator.py`, `scripts/task4_main.py`, `scripts/report_generator.py`
- **Output**: `reports/final_report.pdf`, `reports/task4_insights_report.json`
- **Key Features**:
  - 10-page comprehensive PDF report
  - 6 data visualizations
  - Scenario analysis for all 3 business cases
  - Identification of drivers and pain points
  - Actionable recommendations for each bank
  - Executive summary and conclusions

## 📈 Key Results

### Data Collection
- **Total Reviews**: 1,200+ reviews collected
- **Coverage**: 400+ reviews per bank
- **Time Period**: 6 months of review data
- **Data Quality**: <5% missing data

### Sentiment Analysis
- **Coverage**: 100% of reviews analyzed
- **Accuracy**: Ensemble approach improves reliability
- **Distribution**: Positive/Negative/Neutral breakdown for each bank

### Thematic Insights
- **8 Primary Themes** identified:
  1. Login Issues
  2. Transaction Problems
  3. App Performance
  4. User Interface
  5. Customer Support
  6. Security Concerns
  7. Feature Requests
  8. Account Management

### Database
- **PostgreSQL** schema implemented
- **2 Tables**: banks and reviews
- **Data Integrity**: Foreign key constraints and data validation
- **Scalability**: Supports thousands of reviews

### Final Report
- **10-page PDF** with Medium-style formatting
- **6 Visualizations**:
  1. Rating Distribution by Bank
  2. Sentiment Comparison
  3. Theme Analysis
  4. Word Clouds
  5. Temporal Trends
  6. Recommendations Matrix
- **3 Business Scenarios** analyzed
- **15+ Actionable Recommendations**

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.12+
- PostgreSQL (optional, for Task 3)
- Git

### Setup Instructions

1. **Clone Repository**
```bash
git clone https://github.com/FEBEN-G/bank-reviews-analysis.git
cd bank-reviews-analysis
