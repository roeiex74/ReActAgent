import pandas as pd
import matplotlib.pyplot as plt
import json

try:
    # Load data from CSV
    df = pd.read_csv('whs_2024.csv')

    # Ensure necessary columns are present
    if 'Country name' not in df.columns or 'Healthy life expectancy' not in df.columns:
        raise ValueError("CSV file must contain 'Country name' and 'Healthy life expectancy' columns.")

    # Calculate average healthy life expectancy for all countries
    avg_hle_all = df['Healthy life expectancy'].mean()

    # Filter for top 7 countries
    top_7_countries = ['Iceland', 'Norway', 'Sweden', 'Finland', 'Netherlands', 'Denmark', 'Switzerland']
    df_top_7 = df[df['Country name'].isin(top_7_countries)]
    avg_hle_top_7 = df_top_7['Healthy life expectancy'].mean()

    # Identify highest scoring country
    highest_country = df.loc[df['Healthy life expectancy'].idxmax(), 'Country name']
    highest_hle = df['Healthy life expectancy'].max()

    # Prepare data for plotting
    categories = ['Average All Countries', 'Average Top 7 Countries', highest_country]
    values = [avg_hle_all, avg_hle_top_7, highest_hle]

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color=['blue', 'orange', 'green'])
    plt.ylabel('Healthy Life Expectancy')
    plt.title('Comparison of Healthy Life Expectancy')

    # Save the plot to a file
    plt.savefig('average_hle.png')

    print(json.dumps({'status': 'success', 'message': 'Plot saved as average_hle.png successfully.'}))

except Exception as e:
    print(json.dumps({'status': 'error', 'message': str(e)}))