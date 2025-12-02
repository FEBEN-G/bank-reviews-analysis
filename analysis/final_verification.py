#!/usr/bin/env python3
"""
Final Project Verification
"""
import os
import json
import pandas as pd

def check_task1():
    print("Task 1 Verification:")
    print("-" * 40)
    
    # Check processed file
    if os.path.exists("data/processed/reviews_processed.csv"):
        df = pd.read_csv("data/processed/reviews_processed.csv")
        print(f"✅ Processed data: {len(df):,} reviews")
        
        # Check columns
        required = {'review', 'rating', 'date', 'bank', 'source'}
        if required.issubset(set(df.columns)):
            print("✅ Required columns present")
        else:
            print(f"❌ Missing columns: {required - set(df.columns)}")
            return False
        
        # Check count
        if len(df) >= 1200:
            print("✅ 1200+ reviews: ✓")
        else:
            print(f"❌ Insufficient reviews: {len(df)}")
            return False
        
        # Check per bank
        bank_counts = df['bank'].value_counts()
        for bank in ['CBE', 'BOA', 'Dashen']:
            if bank in bank_counts and bank_counts[bank] >= 400:
                print(f"✅ {bank}: {bank_counts[bank]} reviews")
            else:
                print(f"❌ {bank}: insufficient reviews")
                return False
        
        return True
    else:
        print("❌ Processed file not found")
        return False

def check_task2():
    print("\nTask 2 Verification:")
    print("-" * 40)
    
    files = [
        "data/processed/reviews_with_sentiment.csv",
        "data/processed/reviews_with_themes.csv",
        "reports/task2_final_report.json"
    ]
    
    all_exist = True
    for file in files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            all_exist = False
    
    if all_exist:
        # Check sentiment file
        df = pd.read_csv("data/processed/reviews_with_sentiment.csv")
        if 'sentiment_label' in df.columns:
            print(f"✅ Sentiment analysis: {len(df):,} reviews analyzed")
        else:
            print("❌ Sentiment label missing")
            all_exist = False
    
    return all_exist

def check_task3():
    print("\nTask 3 Verification:")
    print("-" * 40)
    
    files = [
        "database/schema.sql",
        "reports/task3_database_report.json"
    ]
    
    all_exist = True
    for file in files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            all_exist = False
    
    # Check schema content
    if os.path.exists("database/schema.sql"):
        with open("database/schema.sql", "r") as f:
            content = f.read()
            if "CREATE TABLE banks" in content and "CREATE TABLE reviews" in content:
                print("✅ Both tables in schema")
            else:
                print("❌ Tables missing in schema")
                all_exist = False
    
    return all_exist

def main():
    print("="*80)
    print("FINAL PROJECT VERIFICATION")
    print("="*80)
    
    task1_ok = check_task1()
    task2_ok = check_task2()
    task3_ok = check_task3()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if task1_ok and task2_ok and task3_ok:
        print("🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
        print("\n📋 Project ready for submission.")
        print("\nFiles to submit:")
        print("1. All Python scripts in scripts/")
        print("2. All data files in data/processed/")
        print("3. All reports in reports/")
        print("4. Database schema in database/")
        print("5. README.md and requirements.txt")
        
        # Create completion file
        with open("PROJECT_COMPLETED.txt", "w") as f:
            f.write("Customer Experience Analytics Project - COMPLETED ✅\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write("\nTasks Completed:\n")
            f.write("- Task 1: Data Collection & Preprocessing ✓\n")
            f.write("- Task 2: Sentiment & Thematic Analysis ✓\n")
            f.write("- Task 3: PostgreSQL Database Implementation ✓\n")
        
        print(f"\n✅ Completion marker: PROJECT_COMPLETED.txt")
        
        return True
    else:
        print("⚠️  Some tasks need attention.")
        print("\nIssues found in:")
        if not task1_ok: print("  • Task 1")
        if not task2_ok: print("  • Task 2")
        if not task3_ok: print("  • Task 3")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
