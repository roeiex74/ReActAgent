import pandas as pd
import matplotlib.pyplot as plt
import json

try:
    # Load the data from the CSV file
    data = pd.read_csv(
        "whs_2024.csv",
        usecols=[
            "Year",
            "Rank",
            "Country name",
            "Ladder score",
            "upperwhisker",
            "lowerwhisker",
            "Explained by: Log GDP per capita",
            "Explained by: Social support",
            "Explained by: Healthy life expectancy",
            "Explained by: Freedom to make life choices",
            "Explained by: Generosity",
            "Explained by: Perceptions of corruption",
            "Dystopia + residual"
        ],
    )

    # Calculate the average Healthy life expectancy
    average_healthy_life_expectancy = data[
        "Explained by: Healthy life expectancy"
    ].mean()

    # Find the country with the highest Healthy life expectancy
    highest_country = data.loc[
        data["Explained by: Healthy life expectancy"].idxmax()
    ]

    # Prepare the data for plotting
    labels = [
        "Average Healthy Life Expectancy",
        highest_country["Country name"],
    ]
    values = [
        average_healthy_life_expectancy,
        highest_country["Explained by: Healthy life expectancy"],
    ]

    # Plotting
    plt.bar(labels, values, color=["blue", "orange"])
    plt.ylabel("Healthy Life Expectancy")
    plt.title("Average vs Highest Healthy Life Expectancy")

    # Save the plot to a file
    plt.savefig("plot.png")

    # Status response
    print(
        json.dumps({"status": "success", "message": "Plot saved as plot.png"})
    )

except Exception as e:
    # Error response
    print(json.dumps({"status": "error", "message": f"{type(e)}: {str(e)}"}))