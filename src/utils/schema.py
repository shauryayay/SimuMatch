
def validate_schema(df, required_columns, df_name="dataframe"):
    """
    Ensures dataframe has all required columns.
    Fails early with a clear error message.
    """
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"[SCHEMA ERROR] {df_name} missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
