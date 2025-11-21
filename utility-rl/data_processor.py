"""
Data processor for loading and analyzing sales data from Kagle.csv
Handles product segmentation, pricing analysis, and demand modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os


class SalesDataProcessor:
    """Processes sales data for dynamic pricing RL environment"""
    
    def __init__(self, csv_path: str):
        """
        Initialize data processor
        
        Args:
            csv_path: Path to the Kagle.csv file
        """
        self.csv_path = csv_path
        self.raw_data = None
        self.processed_data = None
        self.segments = {}
        self.segment_mapping = {
            'Electronics': 0,
            'Beauty': 1, 
            'Clothing': 2
        }
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the CSV data"""
        try:
            # Load CSV with semicolon delimiter
            self.raw_data = pd.read_csv(self.csv_path, sep=';')
            
            # Clean column names
            self.raw_data.columns = [col.replace('\ufeff', '') for col in self.raw_data.columns]
            
            print(f"Loaded {len(self.raw_data)} transactions")
            print(f"Columns: {list(self.raw_data.columns)}")
            
            return self.raw_data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def process_data(self) -> Dict:
        """
        Process raw data into segments and calculate key metrics
        
        Returns:
            Dictionary with processed data by segment
        """
        if self.raw_data is None:
            self.load_data()
        
        # Convert date column
        self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'], format='%d/%m/%y')
        
        # Convert numeric columns
        numeric_cols = ['Quantity', 'Price per Unit', 'Total Amount', 'Age']
        for col in numeric_cols:
            if col in self.raw_data.columns:
                self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')
        
        # Create product categories if not present
        if 'Product Category' not in self.raw_data.columns:
            # Create dummy categories based on available data
            categories = ['Electronics', 'Beauty', 'Clothing']
            self.raw_data['Product Category'] = np.random.choice(categories, len(self.raw_data))
        
        # Process by segment
        self.processed_data = {}
        
        for segment_name, segment_id in self.segment_mapping.items():
            segment_data = self.raw_data[self.raw_data['Product Category'] == segment_name].copy()
            
            if len(segment_data) > 0:
                # Calculate segment metrics
                avg_price = segment_data['Price per Unit'].mean()
                avg_quantity = segment_data['Quantity'].mean()
                total_sales = segment_data['Total Amount'].sum()
                
                # Calculate cost (85% of price) and utility (15%)
                cost_per_unit = avg_price * 0.85
                utility_per_unit = avg_price * 0.15
                
                # Weekly aggregation
                segment_data['Week'] = segment_data['Date'].dt.isocalendar().week
                weekly_data = segment_data.groupby('Week').agg({
                    'Quantity': 'sum',
                    'Price per Unit': 'mean',
                    'Total Amount': 'sum'
                }).reset_index()
                
                self.processed_data[segment_name] = {
                    'segment_id': segment_id,
                    'avg_price': avg_price,
                    'avg_quantity': avg_quantity,
                    'total_sales': total_sales,
                    'cost_per_unit': cost_per_unit,
                    'utility_per_unit': utility_per_unit,
                    'weekly_data': weekly_data,
                    'reference_price': avg_price,  # template for RL environment
                    'base_demand': avg_quantity,   # template for demand calculation
                    'price_sensitivity': self._get_price_sensitivity(segment_name)
                }
                
                print(f"\n{segment_name} Segment:")
                print(f"  Average Price: ${avg_price:.2f}")
                print(f"  Cost per Unit: ${cost_per_unit:.2f}")
                print(f"  Utility per Unit: ${utility_per_unit:.2f}")
                print(f"  Average Quantity: {avg_quantity:.2f}")
                print(f"  Price Sensitivity: {self.processed_data[segment_name]['price_sensitivity']:.2f}")
                print(f"  Weekly Data Points: {len(weekly_data)}")
        
        return self.processed_data
    
    def _get_price_sensitivity(self, segment_name: str) -> float:
        """
        Get price sensitivity parameter for different segments
        
        Args:
            segment_name: Name of the product segment
            
        Returns:
            Price sensitivity parameter (higher = more sensitive)
        """
        sensitivity_map = {
            'Electronics': 0.5,    # High price sensitivity
            'Beauty': 0.3,         # Medium sensitivity
            'Clothing': 0.4        # Medium-high sensitivity
        }
        return sensitivity_map.get(segment_name, 0.4)
    
    def get_segment_info(self, segment_name: str) -> Dict:
        """
        Get information for a specific segment
        
        Args:
            segment_name: Name of the segment
            
        Returns:
            Dictionary with segment information
        """
        if self.processed_data is None:
            self.process_data()
        
        return self.processed_data.get(segment_name, {})
    
    def get_all_segments(self) -> List[str]:
        """Get list of all available segments"""
        if self.processed_data is None:
            self.process_data()
        
        return list(self.processed_data.keys())
    
    def get_weekly_revenue_data(self, segment_name: str) -> List[float]:
        """
        Get weekly revenue data for a segment (for static pricing baseline)
        
        Args:
            segment_name: Name of the segment
            
        Returns:
            List of weekly revenue values
        """
        if self.processed_data is None:
            self.process_data()
        
        segment_info = self.processed_data.get(segment_name)
        if segment_info:
            weekly_data = segment_info['weekly_data']
            # Calculate revenue for each week (assuming static pricing)
            reference_price = segment_info['reference_price']
            cost_per_unit = segment_info['cost_per_unit']
            
            weekly_revenue = []
            for _, row in weekly_data.iterrows():
                quantity = row['Quantity']
                revenue = (reference_price - cost_per_unit) * quantity
                weekly_revenue.append(revenue)
            
            return weekly_revenue
        
        return []
    
    def export_processed_data(self, output_path: str):
        """
        Export processed data to CSV for analysis
        
        Args:
            output_path: Path to save the processed data
        """
        if self.processed_data is None:
            self.process_data()
        
        # Create summary DataFrame
        summary_data = []
        for segment_name, segment_info in self.processed_data.items():
            summary_data.append({
                'Segment': segment_name,
                'Segment_ID': segment_info['segment_id'],
                'Reference_Price': segment_info['reference_price'],
                'Cost_per_Unit': segment_info['cost_per_unit'],
                'Utility_per_Unit': segment_info['utility_per_unit'],
                'Base_Demand': segment_info['base_demand'],
                'Price_Sensitivity': segment_info['price_sensitivity'],
                'Total_Sales': segment_info['total_sales']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False)
        print(f"Processed data exported to {output_path}")


def main():
    """Test the data processor"""
    # Find the CSV file
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    csv_path = os.path.join(assets_dir, "Kagle.csv")
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        return
    
    # Create processor and load data
    processor = SalesDataProcessor(csv_path)
    processor.load_data()
    processed_data = processor.process_data()
    
    # Export processed data
    output_path = os.path.join(os.path.dirname(__file__), "processed_sales_data.csv")
    processor.export_processed_data(output_path)
    
    print("\nData processing completed successfully!")


if __name__ == "__main__":
    main()
