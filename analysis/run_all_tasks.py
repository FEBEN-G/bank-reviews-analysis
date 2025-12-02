#!/usr/bin/env python3
"""
Complete Project Runner - Runs all tasks for the assignment
"""
import sys
import os
import json
from datetime import datetime

def print_header():
    print("\n" + "="*80)
    print("CUSTOMER EXPERIENCE ANALYTICS FOR FINTECH APPS")
    print("Complete Project Implementation")
    print("="*80)

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("\n🔍 Checking prerequisites...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"  Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("  ⚠️  Python 3.8+ recommended")
    
    # Check required directories
    required_dirs = ['scripts', 'data/raw', 'data/processed', 'reports', 'database']
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"  ✅ Directory exists: {directory}")
        else:
            print(f"  ⚠️  Directory missing: {directory}")
    
    # Check required files
    required_files = ['requirements.txt', 'README.md', 'scripts/task1_main.py']
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ File exists: {file}")
        else:
            print(f"  ❌ File missing: {file}")
    
    print("\n✅ Prerequisites check completed")

def run_task1():
    """Run Task 1"""
    print("\n" + "="*80)
    print("RUNNING TASK 1: DATA COLLECTION AND PREPROCESSING")
    print("="*80)
    
    try:
        sys.path.append('scripts')
        from task1_main import main as task1_main
        
        print("\nStarting Task 1...")
        success = task1_main()
        
        if success:
            print("\n✅ Task 1 completed successfully!")
            return True
        else:
            print("\n❌ Task 1 failed")
            return False
            
    except ImportError as e:
        print(f"\n❌ Error importing task1_main: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error running Task 1: {e}")
        return False

def run_task2():
    """Run Task 2"""
    print("\n" + "="*80)
    print("RUNNING TASK 2: SENTIMENT AND THEMATIC ANALYSIS")
    print("="*80)
    
    try:
        sys.path.append('scripts')
        from task2_main import main as task2_main
        
        print("\nStarting Task 2...")
        success = task2_main()
        
        if success:
            print("\n✅ Task 2 completed successfully!")
            return True
        else:
            print("\n❌ Task 2 failed")
            return False
            
    except ImportError as e:
        print(f"\n❌ Error importing task2_main: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error running Task 2: {e}")
        return False

def run_task3():
    """Run Task 3"""
    print("\n" + "="*80)
    print("RUNNING TASK 3: DATABASE IMPLEMENTATION")
    print("="*80)
    
    try:
        sys.path.append('scripts')
        from task3_main import main as task3_main
        
        print("\nStarting Task 3...")
        success = task3_main()
        
        if success:
            print("\n✅ Task 3 completed successfully!")
            return True
        else:
            print("\n❌ Task 3 failed")
            return False
            
    except ImportError as e:
        print(f"\n❌ Error importing task3_main: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error running Task 3: {e}")
        return False

def generate_final_report():
    """Generate final project report"""
    print("\n" + "="*80)
    print("GENERATING FINAL PROJECT REPORT")
    print("="*80)
    
    try:
        # Collect reports from all tasks
        final_report = {
            'project': 'Customer Experience Analytics for Fintech Apps',
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tasks': {},
            'files_generated': [],
            'requirements_met': {}
        }
        
        # Task 1 report
        task1_report_file = 'reports/task1_kpi_report.json'
        if os.path.exists(task1_report_file):
            with open(task1_report_file, 'r') as f:
                task1_report = json.load(f)
            final_report['tasks']['task1'] = {
                'status': 'COMPLETED',
                'reviews_collected': task1_report.get('total_reviews', 0),
                'data_quality': task1_report.get('missing_data_percentage', 0)
            }
        
        # Task 2 report
        task2_report_file = 'reports/task2_final_report.json'
        if os.path.exists(task2_report_file):
            with open(task2_report_file, 'r') as f:
                task2_report = json.load(f)
            final_report['tasks']['task2'] = {
                'status': 'COMPLETED',
                'sentiment_analysis': True,
                'thematic_analysis': True
            }
        
        # Task 3 report
        task3_report_file = 'reports/task3_database_report.json'
        if os.path.exists(task3_report_file):
            with open(task3_report_file, 'r') as f:
                task3_report = json.load(f)
            final_report['tasks']['task3'] = {
                'status': 'COMPLETED',
                'database_implemented': True
            }
        
        # List all generated files
        generated_files = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith(('.csv', '.json', '.txt', '.md', '.sql')):
                    if 'venv' not in root and '__pycache__' not in root:
                        filepath = os.path.join(root, file)
                        generated_files.append(filepath)
        
        final_report['files_generated'] = sorted(generated_files)[:50]  # Limit to 50
        
        # Check requirements
        requirements = {
            'task1_1200_reviews': final_report['tasks'].get('task1', {}).get('reviews_collected', 0) >= 1200,
            'task1_5_percent_missing': final_report['tasks'].get('task1', {}).get('data_quality', 100) < 5,
            'task2_sentiment_completed': final_report['tasks'].get('task2', {}).get('sentiment_analysis', False),
            'task2_themes_completed': final_report['tasks'].get('task2', {}).get('thematic_analysis', False),
            'task3_database_completed': final_report['tasks'].get('task3', {}).get('database_implemented', False),
            'all_files_generated': len(generated_files) >= 10
        }
        
        final_report['requirements_met'] = requirements
        final_report['all_requirements_met'] = all(requirements.values())
        
        # Save final report
        final_report_file = 'reports/final_project_report.json'
        with open(final_report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"✅ Final report saved: {final_report_file}")
        
        # Generate readable summary
        summary = f"""
{'='*80}
FINAL PROJECT SUMMARY
{'='*80}

Project: Customer Experience Analytics for Fintech Apps
Generated: {final_report['generation_date']}

TASK COMPLETION STATUS:
{'='*80}

Task 1: Data Collection and Preprocessing
  • Status: {final_report['tasks'].get('task1', {}).get('status', 'NOT STARTED')}
  • Reviews Collected: {final_report['tasks'].get('task1', {}).get('reviews_collected', 0):,}
  • Data Quality: {final_report['tasks'].get('task1', {}).get('data_quality', 0)}% missing

Task 2: Sentiment and Thematic Analysis
  • Status: {final_report['tasks'].get('task2', {}).get('status', 'NOT STARTED')}
  • Sentiment Analysis: {'✅ COMPLETED' if final_report['tasks'].get('task2', {}).get('sentiment_analysis') else '❌ NOT COMPLETED'}
  • Thematic Analysis: {'✅ COMPLETED' if final_report['tasks'].get('task2', {}).get('thematic_analysis') else '❌ NOT COMPLETED'}

Task 3: Database Implementation
  • Status: {final_report['tasks'].get('task3', {}).get('status', 'NOT STARTED')}
  • Database: {'✅ IMPLEMENTED' if final_report['tasks'].get('task3', {}).get('database_implemented') else '❌ NOT IMPLEMENTED'}

REQUIREMENTS CHECK:
{'='*80}

"""
        
        for req, met in requirements.items():
            status = "✅ MET" if met else "❌ NOT MET"
            summary += f"{req}: {status}\n"
        
        summary += f"""
OVERALL STATUS: {'✅ ALL REQUIREMENTS MET' if final_report['all_requirements_met'] else '⚠️ SOME REQUIREMENTS NOT MET'}

FILES GENERATED: {len(generated_files):,} files
{'='*80}

Key files:
• data/processed/reviews_processed.csv
• data/processed/reviews_with_sentiment.csv
• data/processed/reviews_with_themes.csv
• reports/task1_kpi_report.json
• reports/task2_final_report.json
• reports/task3_database_report.json
• reports/final_project_report.json
• TASK1_README.md
• README.md

NEXT STEPS FOR SUBMISSION:
{'='*80}

1. Create GitHub repository
2. Create branches for each task (task-1, task-2, task-3)
3. Commit all files with meaningful messages
4. Push to GitHub
5. Submit the repository link

PROJECT READY FOR SUBMISSION: {'✅ YES' if final_report['all_requirements_met'] else '⚠️ NO'}

{'='*80}
"""
        
        # Save summary
        summary_file = 'reports/final_project_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"✅ Project summary saved: {summary_file}")
        
        # Print summary to console
        print(summary)
        
        return final_report['all_requirements_met']
        
    except Exception as e:
        print(f"❌ Error generating final report: {e}")
        return False

def main():
    """Main function"""
    print_header()
    
    print("\n📋 Available Options:")
    print("1. Check Prerequisites")
    print("2. Run Task 1 Only")
    print("3. Run Task 2 Only")
    print("4. Run Task 3 Only")
    print("5. Run All Tasks (Complete Project)")
    print("6. Generate Final Report")
    print("7. Exit")
    
    try:
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            check_prerequisites()
        elif choice == '2':
            run_task1()
        elif choice == '3':
            run_task2()
        elif choice == '4':
            run_task3()
        elif choice == '5':
            print("\n🚀 Running complete project...")
            print("\n" + "="*80)
            
            # Run all tasks
            task1_success = run_task1()
            if task1_success:
                task2_success = run_task2()
                if task2_success:
                    task3_success = run_task3()
                    if task3_success:
                        generate_final_report()
                    else:
                        print("\n❌ Task 3 failed. Project incomplete.")
                else:
                    print("\n❌ Task 2 failed. Project incomplete.")
            else:
                print("\n❌ Task 1 failed. Project incomplete.")
                
        elif choice == '6':
            generate_final_report()
        elif choice == '7':
            print("\nExiting... Goodbye!")
        else:
            print("\nInvalid choice. Please select 1-7.")
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
