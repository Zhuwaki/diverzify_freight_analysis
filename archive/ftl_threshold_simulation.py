
import pandas as pd

def simulate_ftl_thresholds(data, freight_price_col='freight_price', quantity_col='total_quantity', thresholds=[10000, 15000, 20000, 25000]):
    # Ensure numeric columns
    data['quantity'] = pd.to_numeric(data[quantity_col], errors='coerce')
    data['freight_price'] = pd.to_numeric(data[freight_price_col], errors='coerce')

    results = []
    
    for threshold in thresholds:
        # Simulate classification
        data['simulated_type'] = data['quantity'].apply(lambda q: 'FTL' if q >= threshold else 'LTL')
        data['unit_cost'] = data['freight_price'] / data['quantity']
        
        summary = data.groupby('simulated_type')['unit_cost'].agg(['mean', 'count']).reset_index()
        summary['threshold'] = threshold
        results.append(summary)
    
    return pd.concat(results, ignore_index=True)

# Example usage:
# result_df = simulate_ftl_thresholds(vnl_df, 'freight_price', 'total_quantity')
