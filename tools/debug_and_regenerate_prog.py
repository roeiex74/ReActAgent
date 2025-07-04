from tools.openai_client import client, MODEL_4o


def debug_and_regenerate_prog(
    program_fn: str,
    errors: str,
    plot_request: str,
    data_file: str,
    columns: str,
) -> str:

    try:
        with open(program_fn, "r") as file:
            faulty_code = file.read()
    except Exception as e:

        return f"Failed to read program file: {str(e)}"

    ## Reflect on the error and the program

    system_prompt = """You are an expert Python software developer specializing in debugging.
    You're analyzing a failed Python program that was meant to computes a given plot_request and
    store the generated plot in the .png file output_png. 
    The program reads the data from the data_file and uses the columns to compute the plot.
    Analye the received faulty code, reflect on the error message received, plot_request, data_file, columns.
    
    faulty code: The faulty code that was meant to compute the plot.
    error message: The error message received when executing the faulty code.
    plot_request: The plot request we aim to answer when executing the corrected code.
    data_file: The data file received, a csv file.
    columns: The columns received, a comma separated list of columns describing the data in the data_file.
    
    Your task is to produce a detailed error analysis that will guide the next fix attempt.
    
    
    For each error, Return ONLY a JSON object with the following structure:
    {
        "error_summary": "Brief description of the error",
        "error_type": "runtime|logic|output|syntax",
        "error_location": "Specific part of code causing the error",
        "root_cause_analysis": "Explanation of why this error occurred",
        "fix_approach": "Strategy to fix this error"
    }
    If there are multiple errors, return a list of JSON objects.
    if there is no error, return an empty list.
    """

    # Provide the user with information:
    # Plot request, input_file, columns, gen_output_program_fn, output_png and the error message recevied
    user_prompt = f"""Below given: faulty Python program, the error message received, plot_request, data_file, columns.

    ### Faulty code:
    {faulty_code}

    ### Error message:
    {errors}

    ### Plot request:
    {plot_request}

    ### Data file:
    {data_file}

    ### Columns:
    {columns}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_4o,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        reflection = response.choices[0].message.content.strip()

        system_prompt = """You are an expert Python software developer specializing in fixing errors in Python programs.
        You are given a reflection on the error and the faulty program.
        Analyze the reflection details, and fix the program.
        Return ONLY the corrected program.
        IMPORTANT OUTPUT REQUIREMENTS:
        -   Your response MUST be ONLY the complete, corrected Python script.
        -   Do NOT include any explanations, introductory text, or markdown formatting (e.g., ```python ... ```).
        """

        user_prompt = f"""Below given: reflection on the error and the faulty program code.
        
        ### Reflection:
        {reflection}

        ### Faulty program code:
        {faulty_code}
        """

        corrected_code = client.chat.completions.create(
            model=MODEL_4o,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        corrected_code = corrected_code.choices[0].message.content.strip()
        with open(program_fn, "w") as file:
            file.write(corrected_code)

        return "fixed code"
    except Exception as e:
        return f"Failed to regenerate code: {str(e)}"


if __name__ == "__main__":
    program_fn = "plot_program.py"
    errors = "<class 'ValueError'>: Usecols do not match columns, columns expected but not found: ['Failed Column']"
    plot_request = "plot the average 'Healthy life expectancy' for all countries (using 'Country name' column),  and the 'Healthy life expectancy' for the country that has the highest 'Healthy life expectancy' score."
    data_file = "whs_2024.csv"
    columns = "Year,Rank,Country name,Ladder score,upperwhisker,lowerwhisker,Explained by: Log GDP per capita,Explained by: Social support,Explained by: Healthy life expectancy,Explained by: Freedom to make life choices,Explained by: Generosity,Explained by: Perceptions of corruption,Dystopia + residual"
    print(
        debug_and_regenerate_prog(
            program_fn, errors, plot_request, data_file, columns
        )
    )
