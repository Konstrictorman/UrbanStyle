"""
Test script for the dynamic pricing RL system
Tests all components to ensure they work correctly
"""

import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_data_processor():
    """Test the data processor"""
    print("Testing data processor...")
    try:
        from data_processor import SalesDataProcessor
        
        # Find CSV file
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        csv_path = os.path.join(assets_dir, "Kagle.csv")
        
        if not os.path.exists(csv_path):
            print(f"CSV file not found at {csv_path}")
            return False
        
        # Create processor
        processor = SalesDataProcessor(csv_path)
        processor.load_data()
        processed_data = processor.process_data()
        
        print(f"✓ Data processor working - found {len(processed_data)} segments")
        return True
        
    except Exception as e:
        print(f"✗ Data processor error: {e}")
        return False

def test_customer_simulation():
    """Test the customer simulation"""
    print("Testing customer simulation...")
    try:
        from customer_simulation import CustomerSimulation
        
        # Create sample segment info
        segment_info = {
            'reference_price': 100.0,
            'base_demand': 50.0,
            'price_sensitivity': 0.5,
            'cost_per_unit': 85.0,
            'segment_name': 'Electronics'
        }
        
        # Create simulation
        customer_sim = CustomerSimulation(segment_info, random_seed=42)
        
        # Test demand calculation
        demand = customer_sim.calculate_demand(100.0)
        revenue = customer_sim.calculate_revenue(100.0, 85.0)
        
        print(f"✓ Customer simulation working - demand: {demand:.2f}, revenue: ${revenue:.2f}")
        return True
        
    except Exception as e:
        print(f"✗ Customer simulation error: {e}")
        return False

def test_dynamic_pricing_env():
    """Test the dynamic pricing environment"""
    print("Testing dynamic pricing environment...")
    try:
        from dynamic_pricing_env import DynamicPricingEnv
        
        # Create environment
        env = DynamicPricingEnv(segment_name="Electronics")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print(f"✓ Environment reset - observation keys: {list(obs.keys())}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step - reward: {reward:.4f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Dynamic pricing environment error: {e}")
        return False

def test_pricing_agent():
    """Test the pricing agent"""
    print("Testing pricing agent...")
    try:
        from pricing_agent import PricingAgent
        
        # Create agent
        agent = PricingAgent(algorithm="PPO", segment_name="Electronics")
        agent.create_environment()
        agent.create_model()
        
        print("✓ Pricing agent created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Pricing agent error: {e}")
        return False

def test_integration():
    """Test integration between components"""
    print("Testing component integration...")
    try:
        from data_processor import SalesDataProcessor
        from customer_simulation import CustomerSimulation
        from dynamic_pricing_env import DynamicPricingEnv
        
        # Find CSV file
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        csv_path = os.path.join(assets_dir, "Kagle.csv")
        
        if not os.path.exists(csv_path):
            print(f"CSV file not found at {csv_path}")
            return False
        
        # Test full pipeline
        processor = SalesDataProcessor(csv_path)
        processor.load_data()
        processed_data = processor.process_data()
        
        segment_info = processed_data.get("Electronics")
        if not segment_info:
            print("Electronics segment not found")
            return False
        
        # Test environment with real data
        env = DynamicPricingEnv(segment_name="Electronics")
        obs, info = env.reset()
        
        # Run a few steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        
        print("✓ Integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration test error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing Dynamic Pricing RL System")
    print("=" * 50)
    
    tests = [
        test_data_processor,
        test_customer_simulation,
        test_dynamic_pricing_env,
        test_pricing_agent,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nTo run the application:")
        print("python pricing_app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
