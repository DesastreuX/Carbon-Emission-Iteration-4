from itertools import chain

from common.utils import detect_string_variables
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, create_map, lit


def string_columns_encoder(df: DataFrame) -> DataFrame:
    string_columns = detect_string_variables(df)
    for column_name in string_columns:
        mapping = {
            l[column_name]: i
            for i, l in enumerate(df.select(column_name).distinct().collect())
        }
        mapping_expr = create_map([lit(x) for x in chain(*mapping.items())])
        df = df.withColumn(column_name, mapping_expr.getItem(col(column_name)))
    return df
