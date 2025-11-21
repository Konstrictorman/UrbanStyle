"""
Customer simulation for dynamic pricing RL environment
Implements logistic demand function to model realistic customer behavior
"""

import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class CustomerSimulation:
    """
    Simulates customer purchasing behavior using logistic demand function
    Models how customers respond to price changes based on reference prices
    """
    
    def __init__(self, segment_info: Dict, random_seed: int = None):
        """
        Initialize customer simulation for a product segment
        
        Args:
            segment_info: Dictionary containing segment information from data processor
            random_seed: Random seed for reproducible results
        """
        self.segment_info = segment_info
        self.random_seed = random_seed
        
        # Extract parameters from segment info
        self.reference_price = segment_info['reference_price']
        self.base_demand = segment_info['base_demand']
        self.price_sensitivity = segment_info['price_sensitivity']
        self.segment_name = segment_info.get('segment_name', 'Unknown')
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize demand history for tracking
        self.demand_history = []
        self.price_history = []
        
        print(f"Customer simulation initialized for {self.segment_name}")
        print(f"  Reference price: ${self.reference_price:.2f}")
        print(f"  Base demand: {self.base_demand:.2f}")
        print(f"  Price sensitivity: {self.price_sensitivity:.2f}")
    
    def calculate_purchase_probability(self, current_price: float) -> float:
        """
        Calculate purchase probability using logistic function
        
        Args:
            current_price: Current product price
            
        Returns:
            Purchase probability between 0 and 1
        """
        # Logistic demand function
        # P(purchase) = 1 / (1 + exp(-α * (reference_price - current_price)))
        price_difference = self.reference_price - current_price
        exponent = -self.price_sensitivity * price_difference
        purchase_prob = 1 / (1 + np.exp(exponent))
        
        return purchase_prob
    
    def calculate_demand(self, current_price: float, noise_factor: float = 0.1) -> float:
        """
        Calculate actual demand based on current price
        
        Args:
            current_price: Current product price
            noise_factor: Random noise factor (0.1 = 10% variation)
            
        Returns:
            Actual demand (sales volume)
        """
        # Calculate base purchase probability
        purchase_prob = self.calculate_purchase_probability(current_price)
        
        # Calculate base demand
        base_demand = self.base_demand * purchase_prob
        
        # Add random noise to simulate market variability
        noise = np.random.normal(1.0, noise_factor)
        actual_demand = max(0, base_demand * noise)  # Ensure non-negative demand
        
        # Store history for analysis
        self.demand_history.append(actual_demand)
        self.price_history.append(current_price)
        
        return actual_demand
    
    def calculate_revenue(self, current_price: float, cost_per_unit: float) -> float:
        """
        Calculate revenue for given price and cost
        
        Args:
            current_price: Current product price
            cost_per_unit: Cost per unit (from segment info)
            
        Returns:
            Revenue = (price - cost) * demand
        """
        demand = self.calculate_demand(current_price)
        revenue = (current_price - cost_per_unit) * demand
        return revenue
    
    def get_demand_curve(self, price_range: Tuple[float, float] = None, 
                        num_points: int = 100) -> Tuple[List[float], List[float]]:
        """
        Generate demand curve for visualization
        
        Args:
            price_range: Tuple of (min_price, max_price). If None, uses ±50% of reference price
            num_points: Number of points to generate
            
        Returns:
            Tuple of (prices, demands)
        """
        if price_range is None:
            min_price = self.reference_price * 0.5
            max_price = self.reference_price * 1.5
        else:
            min_price, max_price = price_range
        
        prices = np.linspace(min_price, max_price, num_points)
        demands = []
        
        for price in prices:
            # Calculate demand without noise for smooth curve
            purchase_prob = self.calculate_purchase_probability(price)
            demand = self.base_demand * purchase_prob
            demands.append(demand)
        
        return prices.tolist(), demands
    
    def get_revenue_curve(self, price_range: Tuple[float, float] = None, 
                         num_points: int = 100) -> Tuple[List[float], List[float]]:
        """
        Generate revenue curve for visualization
        
        Args:
            price_range: Tuple of (min_price, max_price). If None, uses ±50% of reference price
            num_points: Number of points to generate
            
        Returns:
            Tuple of (prices, revenues)
        """
        if price_range is None:
            min_price = self.reference_price * 0.5
            max_price = self.reference_price * 1.5
        else:
            min_price, max_price = price_range
        
        prices = np.linspace(min_price, max_price, num_points)
        revenues = []
        
        for price in prices:
            revenue = self.calculate_revenue(price, self.segment_info['cost_per_unit'])
            revenues.append(revenue)
        
        return prices.tolist(), revenues
    
    def find_optimal_price(self, cost_per_unit: float, 
                          price_range: Tuple[float, float] = None,
                          precision: float = 0.01) -> Tuple[float, float]:
        """
        Find optimal price that maximizes revenue
        
        Args:
            cost_per_unit: Cost per unit
            price_range: Price range to search in
            precision: Price precision for optimization
            
        Returns:
            Tuple of (optimal_price, max_revenue)
        """
        if price_range is None:
            min_price = cost_per_unit * 1.1  # At least 10% markup
            max_price = self.reference_price * 2.0  # Up to 200% of reference
        else:
            min_price, max_price = price_range
        
        # Grid search for optimal price
        best_price = min_price
        best_revenue = 0
        
        current_price = min_price
        while current_price <= max_price:
            revenue = self.calculate_revenue(current_price, cost_per_unit)
            if revenue > best_revenue:
                best_revenue = revenue
                best_price = current_price
            current_price += precision
        
        return best_price, best_revenue
    
    def plot_demand_curve(self, save_path: str = None):
        """
        Plot demand curve for visualization
        
        Args:
            save_path: Path to save the plot (optional)
        """
        prices, demands = self.get_demand_curve()
        
        plt.figure(figsize=(10, 6))
        plt.plot(prices, demands, 'b-', linewidth=2, label='Demand Curve')
        plt.axvline(self.reference_price, color='r', linestyle='--', 
                   label=f'Reference Price (${self.reference_price:.2f})')
        plt.xlabel('Price ($)')
        plt.ylabel('Demand (Units)')
        plt.title(f'Demand Curve - {self.segment_name} Segment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_revenue_curve(self, save_path: str = None):
        """
        Plot revenue curve for visualization
        
        Args:
            save_path: Path to save the plot (optional)
        """
        prices, revenues = self.get_revenue_curve()
        
        # Find optimal price
        optimal_price, max_revenue = self.find_optimal_price(self.segment_info['cost_per_unit'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(prices, revenues, 'g-', linewidth=2, label='Revenue Curve')
        plt.axvline(optimal_price, color='r', linestyle='--', 
                   label=f'Optimal Price (${optimal_price:.2f})')
        plt.axhline(max_revenue, color='r', linestyle=':', 
                   label=f'Max Revenue (${max_revenue:.2f})')
        plt.xlabel('Price ($)')
        plt.ylabel('Revenue ($)')
        plt.title(f'Revenue Curve - {self.segment_name} Segment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_simulation_stats(self) -> Dict:
        """
        Get statistics from simulation history
        
        Returns:
            Dictionary with simulation statistics
        """
        if not self.demand_history:
            return {}
        
        return {
            'total_episodes': len(self.demand_history),
            'avg_demand': np.mean(self.demand_history),
            'avg_price': np.mean(self.price_history),
            'demand_std': np.std(self.demand_history),
            'price_std': np.std(self.price_history),
            'min_demand': np.min(self.demand_history),
            'max_demand': np.max(self.demand_history)
        }
    
    def reset_history(self):
        """Reset simulation history"""
        self.demand_history = []
        self.price_history = []


def main():
    """Test the customer simulation"""
    # Create sample segment info for testing
    sample_segment_info = {
        'reference_price': 100.0,
        'base_demand': 50.0,
        'price_sensitivity': 0.5,
        'cost_per_unit': 85.0,
        'segment_name': 'Electronics'
    }
    
    # Create customer simulation
    customer_sim = CustomerSimulation(sample_segment_info, random_seed=42)
    
    # Test demand calculation
    test_prices = [80, 90, 100, 110, 120]
    print("\nTesting demand calculation:")
    for price in test_prices:
        demand = customer_sim.calculate_demand(price)
        revenue = customer_sim.calculate_revenue(price, 85.0)
        print(f"Price: ${price:.2f}, Demand: {demand:.2f}, Revenue: ${revenue:.2f}")
    
    # Find optimal price
    optimal_price, max_revenue = customer_sim.find_optimal_price(85.0)
    print(f"\nOptimal price: ${optimal_price:.2f}")
    print(f"Maximum revenue: ${max_revenue:.2f}")
    
    # Plot curves
    customer_sim.plot_demand_curve()
    customer_sim.plot_revenue_curve()


if __name__ == "__main__":
    main()
