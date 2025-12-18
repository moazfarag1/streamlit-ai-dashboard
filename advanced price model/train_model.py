import pandas as pd
import os
from sklearn.model_selection import train_test_split
from model import PriceSalesModel

# --- CONFIGURATION ---
# We use ONLY the 50k file because it contains the real sales logic
SOURCE_FILE = 'retail_sales_dataset_50k.csv' 
MODEL_FILE = 'final_sales_model.pkl'

def main():
    # 1. Load the Real Data
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: {SOURCE_FILE} not found.")
        return
        
    print(f"1. Loading Data ({SOURCE_FILE})...")
    df = pd.read_csv(SOURCE_FILE)
    print(f"   Total rows: {len(df)}")

    # 2. Split into Train (80%) and Test (20%)
    # This ensures we train and test on the SAME Logic/Patterns
    print("2. Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"   Training Set: {len(train_df)} rows")
    print(f"   Testing Set:  {len(test_df)} rows")

    # 3. Train
    print("\n3. Training Model...")
    model = PriceSalesModel()
    model.train(train_df)

    # 4. Evaluate
    # Now the model should perform well because it learned the correct patterns
    model.evaluate(test_df)

    # 5. Save
    model.save(MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()