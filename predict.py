import pandas as pd
import argparse
import os
import sys
from datetime import datetime
from model import PriceSalesModel

# Configuration
MODEL_PATH = 'final_sales_model.pkl'

def get_user_input():
    """Interactive prompts for single product prediction"""
    print("\n--- Enter Product Details ---")
    
    # We set defaults for technical fields to keep the demo quick
    data = {
        'product_id': 'DEMO_001',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'competitor_price': 0, # Will ask below
        'rating': 4.0,         # Default average
        'number_of_reviews': 50,
        'stock_quantity': 100,
        'website_views': 500,
        'cart_additions': 20,
        'return_rate': 0.1,
        'is_holiday_season': 0,
        'is_weekend': 0,
        'days_since_launch': 10,
        'discount_percentage': 0
    }
    
    # Ask for key business drivers
    data['brand'] = input("Brand (e.g., Nike, Zara): ")
    data['category'] = input("Category (e.g., Shoes, Dresses): ")
    data['subcategory'] = input("Subcategory (e.g., Sneakers, Maxi Dresses): ") or "General"
    data['gender'] = input("Gender (Men/Women/Unisex): ")
    data['season'] = input("Season (Summer/Winter/etc): ")
    data['material'] = input("Material (Cotton/Leather/etc): ")
    
    try:
        data['original_price'] = float(input("Original Cost/Price ($): "))
        data['competitor_price'] = float(input("Competitor Price ($): "))
    except ValueError:
        print("Error: Price must be a number.")
        sys.exit(1)
        
    return data

def run_single_mode(model):
    print(">> MODE: Single Product Prediction")
    
    product_data = get_user_input()
    
    print("\nAnalyzing...")
    result = model.predict_single(product_data)
    
    print("\n" + "="*40)
    print("PREDICTION RESULT")
    print("="*40)
    print(f"Product:           {result['brand']} {result['category']}")
    print(f"Recommended Price: ${result['Recommended_Price']:.2f}")
    print(f"Projected Sales:   {result['Predicted_Units_Sold']} units")
    print(f"Projected Revenue: ${result['Predicted_Revenue']:.2f}")
    print("="*40 + "\n")

def run_bulk_mode(model, file_path):
    print(f">> MODE: Bulk Prediction")
    print(f"Reading file: {file_path}")
    
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return

    try:
        input_df = pd.read_csv(file_path)
        print(f"Loaded {len(input_df)} products.")
        
        print("Running predictions...")
        results = model.predict_batch(input_df)
        
        output_filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results.to_csv(output_filename, index=False)
        
        print("\n" + "="*40)
        print("BATCH PROCESS COMPLETE")
        print("="*40)
        print(f"Output saved to: {output_filename}")
        print(f"Total Revenue:   ${results['Predicted_Revenue'].sum():,.2f}")
        print("="*40 + "\n")
        
        # Show Preview
        cols = ['product_id', 'brand', 'Recommended_Price', 'Predicted_Units_Sold']
        print(results[cols].head().to_string(index=False))
        
    except Exception as e:
        print(f"Error processing batch: {e}")

def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description='Retail AI Prediction Tool')
    parser.add_argument('--mode', choices=['single', 'bulk'], required=True, help='Choose "single" for manual input or "bulk" for csv file')
    parser.add_argument('--file', type=str, help='Path to CSV file (required if mode is bulk)')

    args = parser.parse_args()

    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Run train_model.py first.")
        return
    
    model = PriceSalesModel.load(MODEL_PATH)

    # Route logic based on mode
    if args.mode == 'single':
        run_single_mode(model)
    elif args.mode == 'bulk':
        if not args.file:
            print("Error: --file argument is required for bulk mode.")
        else:
            run_bulk_mode(model, args.file)

if __name__ == "__main__":
    main()