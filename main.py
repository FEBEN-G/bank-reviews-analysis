import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_task1():
    """Run Task 1: Data Collection and Preprocessing"""
    print("=" * 60)
    print("TASK 1: DATA COLLECTION AND PREPROCESSING")
    print("=" * 60)
    
    try:
        # Add scripts directory to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
        
        # Import and run task1_main
        from task1_main import main as task1_main
        
        task1_main()
        
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you have installed all requirements:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"Error running Task 1: {e}")
        import traceback
        traceback.print_exc()

def run_task2():
    """Run Task 2: Sentiment and Thematic Analysis"""
    print("=" * 60)
    print("TASK 2: SENTIMENT AND THEMATIC ANALYSIS")
    print("=" * 60)
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
        from task2_main import main as task2_main
        
        task2_main()
    except ImportError as e:
        print(f"Error importing modules: {e}")
    except Exception as e:
        print(f"Error running Task 2: {e}")
        import traceback
        traceback.print_exc()

def run_task3():
    """Run Task 3: Database Implementation"""
    print("=" * 60)
    print("TASK 3: DATABASE IMPLEMENTATION")
    print("=" * 60)
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
        from load_to_database import DataLoader
        
        loader = DataLoader()
        loader.load_all_data()
    except ImportError as e:
        print(f"Error importing modules: {e}")
    except Exception as e:
        print(f"Error running Task 3: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run all tasks"""
    print("=" * 60)
    print("CUSTOMER EXPERIENCE ANALYTICS FOR FINTECH APPS")
    print("=" * 60)
    print("\nAvailable Tasks:")
    print("1. Task 1: Data Collection and Preprocessing")
    print("2. Task 2: Sentiment and Thematic Analysis")
    print("3. Task 3: Database Implementation")
    print("4. Run All Tasks")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nSelect task (1-5): ").strip()
            
            if choice == '1':
                run_task1()
            elif choice == '2':
                run_task2()
            elif choice == '3':
                run_task3()
            elif choice == '4':
                print("\n" + "=" * 60)
                print("RUNNING ALL TASKS")
                print("=" * 60)
                run_task1()
                run_task2()
                run_task3()
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please select 1-5.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
