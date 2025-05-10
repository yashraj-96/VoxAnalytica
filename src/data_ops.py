# src/data_ops.py - Data operations
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import io
import traceback
from contextlib import redirect_stdout

def load_dataset(path):
    if not os.path.exists(path):
        # Create examples directory if it doesn't exist
        os.makedirs('examples', exist_ok=True)
        
        # Create a sample dataset if the requested one doesn't exist
        if 'sales' in path:
            df = create_sample_sales_data()
            df.to_csv(path, index=False)
        elif 'customers' in path:
            df = create_sample_customer_data()
            df.to_csv(path, index=False)
        else:
            raise FileNotFoundError(f"Dataset not found: {path}")
    else:
        df = pd.read_csv(path)
    
    return df

def create_sample_sales_data():
    # Generate dates
    dates = pd.date_range(start='2023-01-01', end='2024-04-30', freq='D')
    
    # Create regions and products
    regions = ['North', 'South', 'East', 'West']
    products = ['Widget A', 'Widget B', 'Widget C', 'Premium X', 'Premium Y']
    
    # Generate random data
    np.random.seed(42)  # For reproducibility
    
    # Create data
    data = []
    for _ in range(5000):
        date = np.random.choice(dates)
        region = np.random.choice(regions)
        product = np.random.choice(products)
        
        # Add seasonality and trends
        base_quantity = 10 + 5 * np.sin(date.month / 12 * 2 * np.pi)
        if 'Premium' in product:
            base_price = 100 + np.random.normal(0, 5)
            base_quantity = base_quantity * 0.5  # Lower volume for premium products
        else:
            base_price = 50 + np.random.normal(0, 3)
        
        # Regional variations
        if region == 'North':
            base_quantity *= 1.2
        elif region == 'South':
            base_price *= 0.9
        
        # Calculate total
        quantity = max(1, int(base_quantity + np.random.normal(0, 3)))
        price = max(10, base_price + np.random.normal(0, 5))
        total = quantity * price
        
        data.append({
            'date': date,
            'region': region,
            'product': product,
            'quantity': quantity,
            'price': round(price, 2),
            'total': round(total, 2)
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add quarter information
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    return df

def create_sample_customer_data():
    # Generate random data
    np.random.seed(42)  # For reproducibility
    
    # Parameters
    n_customers = 1000
    
    # Customer segments
    segments = ['New', 'Regular', 'VIP', 'Inactive']
    segment_probabilities = [0.3, 0.4, 0.1, 0.2]
    
    # Geographic data
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    states = ['NY', 'CA', 'IL', 'TX', 'AZ']
    
    # Create data
    data = []
    for i in range(n_customers):
        # Basic info
        customer_id = f'CUST{i+1000:04d}'
        name = f'Customer {i+1}'
        age = np.random.randint(18, 80)
        
        # Location
        city_idx = np.random.randint(0, len(cities))
        city = cities[city_idx]
        state = states[city_idx]
        
        # Segment
        segment = np.random.choice(segments, p=segment_probabilities)
        
        # Metrics based on segment
        if segment == 'New':
            orders = np.random.randint(1, 5)
            retention_prob = 0.5 + np.random.normal(0, 0.1)
            last_order_days = np.random.randint(1, 30)
        elif segment == 'Regular':
            orders = np.random.randint(5, 20)
            retention_prob = 0.7 + np.random.normal(0, 0.1)
            last_order_days = np.random.randint(10, 90)
        elif segment == 'VIP':
            orders = np.random.randint(20, 100)
            retention_prob = 0.9 + np.random.normal(0, 0.05)
            last_order_days = np.random.randint(1, 45)
        else:  # Inactive
            orders = np.random.randint(1, 10)
            retention_prob = 0.2 + np.random.normal(0, 0.1)
            last_order_days = np.random.randint(180, 365)
        
        # Financial metrics
        avg_order_value = np.random.normal(50, 20)
        if segment == 'VIP':
            avg_order_value = np.random.normal(200, 50)
        
        total_spent = orders * avg_order_value
        
        # Satisfaction
        satisfaction = np.random.normal(0.7, 0.15)
        if segment == 'VIP':
            satisfaction = np.random.normal(0.9, 0.05)
        elif segment == 'Inactive':
            satisfaction = np.random.normal(0.5, 0.2)
        
        # Email engagement
        email_open_rate = np.random.normal(0.3, 0.1)
        if segment == 'VIP':
            email_open_rate = np.random.normal(0.6, 0.1)
        
        data.append({
            'customer_id': customer_id,
            'name': name,
            'age': age,
            'city': city,
            'state': state,
            'segment': segment,
            'total_orders': orders,
            'avg_order_value': round(avg_order_value, 2),
            'total_spent': round(total_spent, 2),
            'retention_probability': round(min(1, max(0, retention_prob)), 2),
            'days_since_last_order': last_order_days,
            'satisfaction_score': round(min(1, max(0, satisfaction)), 2),
            'email_open_rate': round(min(1, max(0, email_open_rate)), 2)
        })
    
    return pd.DataFrame(data)

def execute_code(code, df):
    # Set up the execution environment
    local_vars = {
        'df': df,
        'pd': pd,
        'np': np,
        'plt': plt,
        'stats': stats,
        'result': None
    }
    
    # Redirect stdout to capture print statements
    f = io.StringIO()
    
    try:
        with redirect_stdout(f):
            # Execute the code
            exec(code, local_vars)
        
        # Get the matplotlib figure if one was created
        fig = None
        for obj in local_vars.values():
            if isinstance(obj, plt.Figure):
                fig = obj
                break
        
        # If no figure was explicitly created but plt was used,
        # capture the current figure
        if fig is None and plt.get_fignums():
            fig = plt.gcf()
        
        # Get the result variable
        result = local_vars.get('result')
        
        # If there's no result but there's console output, use that
        if result is None and f.getvalue().strip():
            result = f.getvalue().strip()
        
        return result, fig
    
    except Exception as e:
        error_msg = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
        return error_msg, None
    

if __name__=='__main__':
    df = create_sample_sales_data() 
    print(df)