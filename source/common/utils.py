from functools import reduce

from common.z_score import abs_z_score
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import IntegerType, NumericType


def describe_dataframe_details(spark: SparkSession, df: DataFrame) -> DataFrame:
    """
    Generate a summary table with details about a DataFrame.

    Args:
        spark (SparkSession): The SparkSession object.
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The summary table with details about the DataFrame.
    """
    # Detect continuous variables in the DataFrame
    numeric_columns = detect_continuous_variables(df=df)

    # Calculate the absolute z-scores for the continuous variables
    abs_z_score_df = abs_z_score(df=df, columns=numeric_columns)

    # Count the number of outliers for each column
    outlier_counts = ["outlier_count"] + [
        abs_z_score_df.select(col_name).where(abs_z_score_df[col_name] >= 3).count()
        for col_name in abs_z_score_df.columns
    ]

    # Create a DataFrame with the outlier counts
    column_names = abs_z_score_df.columns
    column_names.insert(0, "summary")
    outlier_counts_df = spark.createDataFrame([tuple(outlier_counts)], column_names)

    # Generate the summary table using the DataFrame's describe() method
    info_table = df.describe()

    # Union the outlier counts DataFrame with the summary table
    info_table = (
        info_table.unionByName(outlier_counts_df, allowMissingColumns=True).toPandas().T
    )

    # Set the column names of the summary table
    info_table.columns = info_table.iloc[0]
    info_table.drop(info_table.index[0], inplace=True)

    # Calculate the null count for each column
    info_table["null_count"] = df.count() - info_table["count"].astype(int)

    return info_table


def detect_continuous_variables(df: DataFrame) -> list[str]:
    """
    Detects continuous variables in a DataFrame.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        list[str]: A list of column names that are continuous variables.
    """
    # Initialize an empty list to store the continuous column names
    continuous_columns = []

    # Iterate over each column in the DataFrame
    for column_name in df.columns:
        # Get the data type of the column
        dtype = df.schema[column_name].dataType

        # Check if the data type is IntegerType or NumericType
        if isinstance(dtype, (IntegerType, NumericType)):
            # If it is, add the column name to the list of continuous columns
            continuous_columns.append(column_name)

    # Return the list of continuous column names
    return continuous_columns


def change_case(string: str):
    """
    Changes the case of a given string to snake case by replacing spaces with underscores and upper case to lower case.

    Parameters:
    string (str): The input string to be modified.

    Returns:
    str: The modified string with case changed and spaces replaced by underscores.
    """
    # Using reduce to concatenate characters and add underscores before uppercase letters
    modified_string = reduce(lambda x, y: x + ("_" if y.isupper() else "") + y, string)
    # Replacing spaces with underscores and removing any double underscores
    final_string = modified_string.replace(" ", "_").replace("__", "_").lower()

    return final_string


def rename_columns(df: DataFrame)->DataFrame:
    rename_mapping = {
        column_name: change_case(string=column_name) for column_name in df.columns
    }
    for k, v in rename_mapping.items():
        if k != v:
            df = df.withColumnRenamed(k, v)
    return df