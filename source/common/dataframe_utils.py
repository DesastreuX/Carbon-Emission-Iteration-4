from itertools import chain

from common.utils import detect_array_variables, detect_string_variables
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, create_map, lit


def string_columns_encoder(df: DataFrame) -> DataFrame:
    """
    Encodes string columns in a DataFrame using String Indexer.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The DataFrame with string columns encoded.
    """
    # Detect string columns
    string_columns = detect_string_variables(df)

    # Create a list of output column names
    output_columns = [f"{column}_index" for column in string_columns]

    # Create a string indexer for the column
    string_indexer = StringIndexer(inputCols=string_columns, outputCols=output_columns)

    # Fit and transform the DataFrame using the string columns pipeline
    encoded_df = string_indexer.fit(df).transform(df)

    # Drop the original string columns
    encoded_df = encoded_df.drop(*string_columns)

    # Rename the output columns to match the original string column names
    for old_name, new_name in zip(output_columns, string_columns):
        encoded_df = encoded_df.withColumnRenamed(old_name, new_name)

    return encoded_df


def array_to_string(df: DataFrame) -> DataFrame:
    """
    Convert array columns to string columns in a DataFrame.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The DataFrame with array columns converted to string columns.
    """
    # Detect array columns
    array_columns = detect_array_variables(df)

    # Convert array columns to string columns
    for column_name in array_columns:
        df = df.withColumn(column_name, col(column_name).cast("string"))

    return df
