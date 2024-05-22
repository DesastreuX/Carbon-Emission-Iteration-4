from pandas.core.frame import DataFrame
from pyspark.sql.functions import abs, col, mean, stddev


def z_score(df: DataFrame, columns: list[str]) -> DataFrame:
    """
    Calculate the z-scores for the specified columns in a PySpark DataFrame.

    Parameters:
        df (pyspark.sql.DataFrame): The input DataFrame.
        columns (list[str]): The list of column names for which to calculate the z-scores.

    Returns:
        pyspark.sql.DataFrame: The DataFrame with the z-scores added as new columns.

    Example:
        df = spark.createDataFrame([(1, 2.0), (2, 3.0), (3, 4.0)], ["id", "value"])
        z_score(df, ["value"])
        +---+-----+
        |id |value|
        +---+-----+
        |  1|  -1.0|
        |  2|   0.0|
        |  3|   1.0|
        +---+-----+
    """
    # Create an alias for the input DataFrame
    z_score_df = df.alias("z_score_df")

    for column in columns:
        # Calculate the mean and standard deviation for the current column
        stats = df.select(
            mean(col(column)).alias("mean"), stddev(col(column)).alias("stddev")
        ).collect()[0]

        # Calculate the z-score for the current column
        z_score_df = z_score_df.withColumn(
            column, (df[column] - stats["mean"]) / stats["stddev"]
        )

    # Add the id column if it exists
    if "id" in df.columns:
        columns.append("id")

    # Select only the specified columns from the z-scored DataFrame
    return z_score_df.select(columns)


def abs_z_score(df: DataFrame, columns: list[str]) -> DataFrame:
    """
    Calculate the absolute z-scores for the specified columns in a PySpark DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.
        columns (list[str]): The list of column names for which to calculate the absolute z-scores.

    Returns:
        DataFrame: The DataFrame with the absolute z-scores added as new columns.

    Example:
        df = spark.createDataFrame([(1, 2.0), (2, 3.0), (3, 4.0)], ["id", "value"])
        abs_z_score(df, ["value"])
        +---+--------+
        |id |value   |
        +---+--------+
        |  1|  1.0   |
        |  2|  1.0   |
        |  3|  1.0   |
        +---+--------+
    """
    # Calculate the z-scores for the specified columns
    z_score_df = z_score(df=df, columns=columns)

    # Calculate the absolute z-scores by taking the absolute value of each z-score column
    for column in columns:
        z_score_df = z_score_df.withColumn(f"{column}", abs(col(column)))

    return z_score_df
