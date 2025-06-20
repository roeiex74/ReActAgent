from openai import AzureOpenAI
import json
from dotenv import load_dotenv
import os

load_dotenv()


# Model and version
MODEL_4o = os.getenv("MODEL_4o")
AZURE_OPEN_VERSION_4o = os.getenv("AZURE_OPEN_VERSION_4o")

# Read Azure credentials
AZURE_OPENAI_API_KEY = os.getenv("CLASS_OPEN_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("SUBSCRIPTION_OPENAI_ENDPOINT")

# Initialize the OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPEN_VERSION_4o,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)


def gen_plot_prog(
    plot_request: str,
    input_file: str,
    columns: str,
    gen_output_program_fn: str,
    output_png: str,
) -> str:
    """
    Generates a Python plotting script from a natural language request.

    Args:
        plot_request: Description of the plot to generate.
        input_file: Path to the CSV file to analyze.
        columns: Comma-separated list of column names.
        gen_output_program_fn: Path to write the generated Python script.
        output_png: Path to save the resulting .png plot.

    Returns:
        str: The full code as a string (also written to gen_output_program_fn)
    """

    try:
        system_prompt = (
            "You are a Python code generator that writes plotting code using matplotlib and pandas. You are an expert in data visualization and analysis.\n"
            "The generated program should:\n"
            f"- Read from the CSV file: {input_file}\n"
            f"- Use only the following columns: {columns}\n"
            f"- Save the plot to: {output_png}\n"
            "- DO NOT include plt.show().\n"
            f"- Use plt.savefig('{output_png}') instead to save and exit cleanly.\n"
            "- Return only the code."
            "- The code should be able to handle exceptions and errors gracefully."
            "- Analyze the plot request, formalize it in data analysis terms, and continue the task."
            "The script should print ONLY the Python code to standard output, without any introductory text, explanations, or markdown formatting like ```python ... ```."
            "NOTES: \nCSV File contains headers, and the first row is the header row. The header row is the column names. The data starts from the second row.\nThe Script should be able to provide a status response to the user in case of a error or exception or a success - in the following JSON format: {status: 'success' | 'error', message: 'success message' | 'error message/ exception message'}"
        )

        user_prompt = f"Plot Request: {plot_request}"

        response = client.chat.completions.create(
            model=MODEL_4o,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        code = response.choices[0].message.content.strip()

        # Optional check
        if "plt.show" in code and "plt.savefig" not in code:
            return json.dumps(
                {
                    "error": "Generated code uses plt.show() instead of plt.savefig().",
                    "code": code,
                }
            )

        # Write to output file
        with open(gen_output_program_fn, "w", encoding="utf-8") as f:
            f.write(code)

        return code

    except Exception as e:
        return json.dumps(
            {
                "error": str(e),
                "status": "failed",
                "output_file": gen_output_program_fn,
            }
        )


if __name__ == "__main__":
    plot_request = "plot the average 'Healthy life expectancy' for all countries (using 'Country name' column) combined,  and the 'Healthy life expectancy' for the country that has the highest 'Healthy life expectancy' score."
    input_file = "whs_2024.csv"
    columns = "Year,Rank,Country name,Ladder score,upperwhisker,lowerwhisker,Explained by: Log GDP per capita,Explained by: Social support,Explained by: Healthy life expectancy,Explained by: Freedom to make life choices,Explained by: Generosity,Explained by: Perceptions of corruption,Dystopia + residual"
    gen_output_program_fn = "plot_program.py"
    output_png = "plot.png"
    print(
        gen_plot_prog(
            plot_request,
            input_file,
            columns,
            gen_output_program_fn,
            output_png,
        )
    )
