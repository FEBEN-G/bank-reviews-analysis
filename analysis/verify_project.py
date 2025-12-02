#!/usr/bin/env python3
"""
Final project verification
"""
import os
import json
import pandas as pd

def verify_project():
    print("Final Project Verification")
    print("="*60)
    
    all_good = True
    issues = []
    
    # Check Task 1
    print("\n📋 Task 1 Verification:")
    
    # Check if processed file exists
    processed_file = 'data/processed/reviews_processed.csv'
    if os.path.exists(processed_file):
        try:
            df = pd.read_csv(processed_file)
            print(f"  ✅ Processed data: {len(df):,} reviews")
            
            # Check columns
            required_columns = {'review', 'rating', 'date', 'bank', 'source'}
            if required_columns.issubset(set(df.columns)):
                print("  ✅ Required columns present")
            else:
                print(f"  ❌ Missing columns: {required_columns - set(df.columns)}")
                issues.append("Missing required columns in processed data")
                all_good = False
            
            # Check review count
            if len(df) >= 1200:
                print(f"  ✅ 1200+ reviews: {len(df):,}")
            else:
                print(f"  ❌ Insufficient reviews: {len(df):,}")
                issues.append(f"Insufficient reviews: {len(df)}")
                all_good = False
                
        except Exception as e:
            print(f"  ❌ Error reading processed file: {e}")
            issues.append(f"Error reading processed file: {e}")
            all_good = False
    else:
        print(f"  ❌ Processed file not found: {processed_file}")
        issues.append("Processed data file not found")
        all_good = False
    
    # Check Task 2
    print("\n📊 Task 2 Verification:")
    
    sentiment_file = 'data/processed/reviews_with_sentiment.csv'
    theme_file = 'data/processed/reviews_with_themes.csv'
    
    if os.path.exists(sentiment_file):
        try:
            df_sentiment = pd.read_csv(sentiment_file)
            print(f"  ✅ Sentiment data: {len(df_sentiment):,} reviews")
            
            if 'sentiment_label' in df_sentiment.columns:
                sentiment_counts = df_sentiment['sentiment_label'].value_counts()
                print(f"  ✅ Sentiment analysis: {sentiment_counts.to_dict()}")
            else:
                print("  ❌ Sentiment label column missing")
                issues.append("Sentiment label column missing")
                all_good = False
                
        except Exception as e:
            print(f"  ❌ Error reading sentiment file: {e}")
            issues.append(f"Error reading sentiment file: {e}")
            all_good = False
    else:
        print(f"  ❌ Sentiment file not found: {sentiment_file}")
        issues.append("Sentiment data file not found")
        all_good = False
    
    if os.path.exists(theme_file):
        try:
            df_theme = pd.read_csv(theme_file)
            print(f"  ✅ Theme data: {len(df_theme):,} reviews")
            
            if 'themes' in df_theme.columns:
                print("  ✅ Themes column present")
            else:
                print("  ❌ Themes column missing")
                issues.append("Themes column missing")
                all_good = False
                
        except Exception as e:
            print(f"  ❌ Error reading theme file: {e}")
            issues.append(f"Error reading theme file: {e}")
            all_good = False
    else:
        print(f"  ❌ Theme file not found: {theme_file}")
        issues.append("Theme data file not found")
        all_good = False
    
    # Check reports
    print("\n📄 Report Verification:")
    
    report_files = [
        'reports/task1_kpi_report.json',
        'reports/task2_final_report.json',
        'reports/task3_database_report.json'
    ]
    
    for report_file in report_files:
        if os.path.exists(report_file):
            print(f"  ✅ {report_file}")
        else:
            print(f"  ❌ {report_file} - MISSING")
            issues.append(f"Report file missing: {report_file}")
            all_good = False
    
    # Check documentation
    print("\n📘 Documentation Verification:")
    
    doc_files = ['README.md', 'TASK1_README.md']
    for doc_file in doc_files:
        if os.path.exists(doc_file):
            print(f"  ✅ {doc_file}")
        else:
            print(f"  ❌ {doc_file} - MISSING")
            issues.append(f"Documentation missing: {doc_file}")
            all_good = False
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL VERIFICATION SUMMARY")
    print("="*60)
    
    if all_good:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\n✅ Project is ready for submission")
        print("\nNext steps:")
        print("1. Create GitHub repository")
        print("2. Commit all files")
        print("3. Submit repository link")
    else:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  • {issue}")
        
        print("\n❌ Please fix the issues above before submission")
    
    return all_good

if __name__ == "__main__":
    success = verify_project()
    exit(0 if success else 1)
