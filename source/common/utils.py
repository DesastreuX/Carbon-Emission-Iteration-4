from functools import reduce

import matplotlib.pyplot as plt
import seaborn as sns
from common.z_score import abs_z_score
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import sum
from pyspark.sql.types import ArrayType, IntegerType, NumericType, StringType


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


def detect_string_variables(df: DataFrame) -> list[str]:
    """
    Detects string variables in a DataFrame.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        list[str]: A list of column names that are string variables.
    """
    string_columns = []

    # Iterate over each column in the DataFrame
    for column_name in df.columns:
        # Get the data type of the column
        dtype = df.schema[column_name].dataType

        # Check if the data type is StringType
        if isinstance(dtype, StringType):
            # If it is, add the column name to the list of string columns
            string_columns.append(column_name)

    return string_columns


def detect_array_variables(df: DataFrame) -> list[str]:
    """
    Detects array variables in a DataFrame.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        list[str]: A list of column names that are array variables.
    """
    array_columns = []

    # Iterate over each column in the DataFrame
    for column_name in df.columns:
        # Get the data type of the column
        dtype = df.schema[column_name].dataType

        # Check if the data type is ArrayType
        if isinstance(dtype, ArrayType):
            # If it is, add the column name to the list of array columns
            array_columns.append(column_name)

    return array_columns


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


def rename_columns(df: DataFrame) -> DataFrame:
    """
    Renames the columns of a DataFrame by converting them to snake case.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The DataFrame with renamed columns.
    """
    # Create a dictionary mapping the old column names to the new column names in snake case
    rename_mapping = {
        column_name: change_case(
            string=column_name
        )  # Convert column name to snake case
        for column_name in df.columns
    }

    # Rename the columns in the DataFrame
    for old_name, new_name in rename_mapping.items():
        if (
            old_name != new_name
        ):  # Only rename if the new name is different from the old name
            df = df.withColumnRenamed(old_name, new_name)

    return df


def plt_group_by_2_columns_and_sum(df: DataFrame, group_by: list[str], sum_column: str):
    """
    Plot a bar plot with group by columns and sum of a specific column.

    Args:
        df (DataFrame): The input DataFrame.
        group_by (list[str]): The list of columns to group by.
        sum_column (str): The column to calculate the sum of.

    Returns:
        None
    """
    # Calculate the sum of the sum_column for group_by columns
    df_grouped = (
        df.groupBy(group_by)
        .agg(sum(sum_column).alias(f"sum_{sum_column}"))
        .sort(group_by)
    )

    # Convert the grouped DataFrame to a Pandas DataFrame
    plot_df = df_grouped.toPandas()

    # Set the seaborn theme
    sns.set_theme(context="talk")

    # Create a figure with the specified size
    plt.figure(figsize=(11, 8))

    # Plot the bar plot with the x-axis as the first group_by column,
    # the y-axis as the sum of the sum_column, and the hue as the second group_by column
    sns.barplot(x=group_by[0], y=f"sum_{sum_column}", hue=group_by[1], data=plot_df)

    # Set the title of the plot
    plt.title(f"{' and '.join(group_by)} with sum_{sum_column} plot", y=1.05)

    # Add a legend to the plot
    plt.legend(loc="upper right")


def scatter_plot(df: DataFrame, x: str, y: str, hue: str):
    """
    Plot a scatter plot with the specified x, y, and hue variables.

    Args:
        df (DataFrame): The input DataFrame.
        x (str): The name of the x-axis variable.
        y (str): The name of the y-axis variable.
        hue (str): The name of the hue variable.

    Returns:
        None
    """
    # Convert the DataFrame to a Pandas DataFrame
    plot_df = df.toPandas()

    # Set the seaborn theme
    sns.set_theme(context="talk")

    # Create a figure with the specified size
    plt.figure(figsize=(11, 8))

    # Plot the scatter plot with the specified x, y, and hue variables
    sns.scatterplot(x=x, y=y, hue=hue, data=plot_df)

    # Set the title of the plot
    plt.title(f"{x} and {y} plot", y=1.05)

    # Add a legend to the plot
    plt.legend(loc="upper right")


def plot_column_value_count(df: DataFrame, column_name: str):
    """
    Plot a bar plot with the count of values in a given column.

    Args:
        df (DataFrame): The input DataFrame.
        column_name (str): The name of the column to count values of.

    Returns:
        None
    """
    # Group the DataFrame by the specified column and count the number of occurrences of each value
    plot_df = df.groupBy(column_name).count().toPandas()

    # Set the seaborn theme
    sns.set_theme(context="talk")

    # Create a figure with the specified size
    plt.figure(figsize=(11, 8))

    # Plot a bar plot with the x-axis as the values of the specified column,
    # the y-axis as the count of occurrences, and the hue as the values of the specified column
    sns.barplot(y=column_name, x="count", hue=column_name, data=plot_df)

    # Set the title of the plot
    plt.title(f"Count of Value in {column_name} plot", y=1.05)

    # Add a legend to the plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
