import pandas as pd
from pathlib import Path


# ============================================================
# IO-VNBD DATA LOADER
# ============================================================

# Path to the IO-VNBD smartphone dataset
DATA_PATH = Path(
    "data/raw/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)


def load_io_vnbd(file_path=DATA_PATH):
    """
    Load the IO-VNBD smartphone dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned smartphone sensor and GNSS data.
    """

    print("Loading dataset...")
    print(f"File: {file_path}")

    # Read CSV
    df = pd.read_csv(file_path, encoding="cp1252")

    print(f"\nDataset loaded successfully.")
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    return df


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):
    """
    Remove unnecessary spaces from column names.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
    )

    return df


# ============================================================
# PREPARE DATA TYPES
# ============================================================

def prepare_data(df):
    """
    Convert timestamps and numerical sensor columns
    into appropriate data types.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Convert date/time
    # --------------------------------------------------------

    date_column = "DATE_YYYY-MO-DD_HH-MI-SS_SSS"

    if date_column in df.columns:

        df["timestamp"] = pd.to_datetime(
            df[date_column],
            format="%Y-%m-%d %H:%M:%S:%f",
            errors="coerce"
        )

    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    for column in df.columns:

        # Do not convert date/time columns
        if column not in [date_column, "timestamp"]:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    df = load_io_vnbd()

    print("\nOriginal columns:")
    print(df.columns.tolist())

    df = clean_column_names(df)

    print("\nCleaned columns:")
    print(df.columns.tolist())

    df = prepare_data(df)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())