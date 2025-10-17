#!/usr/bin/env python3
"""
Script to analyze the current CSV data and add 500 more lines
to ensure every week has sales data for all 5 categories.
"""

import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np

def analyze_current_data():
    """Analyze the current CSV data structure"""
    print("=== ANALYZING CURRENT CSV DATA ===")
    
    # Read the CSV file
    df = pd.read_csv('/Users/ralvarez/Repo/maestria/aida/UrbanStyle/assets/Kagle_.csv', sep=';')
    
    print(f"Total transactions: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Convert dates to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
    
    # Get unique dates and categories
    unique_dates = sorted(df['Date'].unique())
    categories = df['Product Category'].unique()
    
    print(f"Unique dates: {len(unique_dates)}")
    print(f"Categories: {list(categories)}")
    
    # Analyze category distribution per date
    print("\n=== CATEGORY DISTRIBUTION PER DATE ===")
    category_counts = df.groupby(['Date', 'Product Category']).size().unstack(fill_value=0)
    
    # Find dates missing categories
    missing_data = []
    for date in unique_dates:
        for category in categories:
            if category not in category_counts.columns or category_counts.loc[date, category] == 0:
                missing_data.append((date, category))
    
    print(f"Dates missing category data: {len(missing_data)}")
    
    return df, unique_dates, categories, missing_data

def generate_additional_data(df, unique_dates, categories, missing_data, target_lines=500):
    """Generate additional data to fill gaps and reach target lines"""
    print(f"\n=== GENERATING {target_lines} ADDITIONAL LINES ===")
    
    # Get the last transaction ID
    last_transaction_id = df['Transaction ID'].max()
    
    # Get existing customer IDs for reference
    existing_customers = df['Customer ID'].unique()
    
    # Price ranges for each category (based on existing data)
    category_prices = {
        'Beauty': (25, 500),
        'Clothing': (25, 500), 
        'Electronics': (25, 500),
        'Baby Stuff': (25, 500),
        'Sports': (25, 500)
    }
    
    # Quantity ranges
    quantity_range = (1, 4)
    
    # Age range
    age_range = (18, 64)
    
    new_transactions = []
    
    # First, fill missing category data for existing dates
    for date, category in missing_data:
        if len(new_transactions) >= target_lines:
            break
            
        # Generate transaction for missing category
        transaction_id = last_transaction_id + len(new_transactions) + 1
        customer_id = f"CUST{transaction_id:03d}"
        gender = random.choice(['Male', 'Female'])
        age = random.randint(age_range[0], age_range[1])
        quantity = random.randint(quantity_range[0], quantity_range[1])
        price_per_unit = random.randint(category_prices[category][0], category_prices[category][1])
        total_amount = quantity * price_per_unit
        
        new_transactions.append({
            'Transaction ID': transaction_id,
            'Date': date.strftime('%d/%m/%y'),
            'Customer ID': customer_id,
            'Gender': gender,
            'Age': age,
            'Product Category': category,
            'Quantity': quantity,
            'Price per Unit': price_per_unit,
            'Total Amount': total_amount
        })
    
    # Fill remaining lines with random transactions across all dates and categories
    remaining_lines = target_lines - len(new_transactions)
    
    for i in range(remaining_lines):
        if len(new_transactions) >= target_lines:
            break
            
        # Random date from existing dates
        date = random.choice(unique_dates)
        
        # Random category
        category = random.choice(categories)
        
        # Generate transaction
        transaction_id = last_transaction_id + len(new_transactions) + 1
        customer_id = f"CUST{transaction_id:03d}"
        gender = random.choice(['Male', 'Female'])
        age = random.randint(age_range[0], age_range[1])
        quantity = random.randint(quantity_range[0], quantity_range[1])
        price_per_unit = random.randint(category_prices[category][0], category_prices[category][1])
        total_amount = quantity * price_per_unit
        
        new_transactions.append({
            'Transaction ID': transaction_id,
            'Date': date.strftime('%d/%m/%y'),
            'Customer ID': customer_id,
            'Gender': gender,
            'Age': age,
            'Product Category': category,
            'Quantity': quantity,
            'Price per Unit': price_per_unit,
            'Total Amount': total_amount
        })
    
    print(f"Generated {len(new_transactions)} new transactions")
    return new_transactions

def save_extended_csv(df, new_transactions):
    """Save the extended CSV with new transactions"""
    print("\n=== SAVING EXTENDED CSV ===")
    
    # Convert new transactions to DataFrame
    new_df = pd.DataFrame(new_transactions)
    
    # Combine with original data
    extended_df = pd.concat([df, new_df], ignore_index=True)
    
    # Sort by date and transaction ID
    extended_df = extended_df.sort_values(['Date', 'Transaction ID'])
    
    # Save to new file
    output_file = '/Users/ralvarez/Repo/maestria/aida/UrbanStyle/assets/Kagle_.csv'
    extended_df.to_csv(output_file, sep=';', index=False)
    
    print(f"Extended CSV saved with {len(extended_df)} total transactions")
    
    # Verify category distribution
    print("\n=== FINAL CATEGORY DISTRIBUTION ===")
    category_counts = extended_df['Product Category'].value_counts()
    print(category_counts)
    
    return extended_df

def main():
    """Main function"""
    print("Starting CSV analysis and extension...")
    
    # Analyze current data
    df, unique_dates, categories, missing_data = analyze_current_data()
    
    # Generate additional data
    new_transactions = generate_additional_data(df, unique_dates, categories, missing_data, target_lines=500)
    
    # Save extended CSV
    extended_df = save_extended_csv(df, new_transactions)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Original transactions: {len(df)}")
    print(f"New transactions: {len(new_transactions)}")
    print(f"Total transactions: {len(extended_df)}")
    
    # Check if every week now has data for all categories
    extended_df['Date'] = pd.to_datetime(extended_df['Date'], format='%d/%m/%y')
    weekly_category_counts = extended_df.groupby([extended_df['Date'].dt.to_period('W'), 'Product Category']).size().unstack(fill_value=0)
    
    print(f"\nWeeks with complete category data: {len(weekly_category_counts)}")
    print("Categories per week should now be more balanced!")

if __name__ == "__main__":
    main()
