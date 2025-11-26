"""Data loading utilities for the research project."""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def load_csv(table_name: str) -> pd.DataFrame:
    """Load a CSV file by table name."""
    file_path = DATA_DIR / f"{table_name}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Table {table_name} not found at {file_path}")
    
    logger.debug(f"Loading CSV: {file_path}")
    return pd.read_csv(file_path)


def load_all_tables() -> dict[str, pd.DataFrame]:
    """Load all CSV tables into a dictionary."""
    tables = {}
    for csv_file in DATA_DIR.glob("*.csv"):
        table_name = csv_file.stem
        tables[table_name] = pd.read_csv(csv_file)
        logger.debug(f"Loaded table: {table_name} ({len(tables[table_name])} rows)")
    return tables


def load_metadata() -> dict:
    """Load the schema metadata JSON."""
    metadata_path = DATA_DIR / "schema_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    with open(metadata_path) as f:
        return json.load(f)


def get_csv_as_string(table_name: str) -> str:
    """Get raw CSV content as a string."""
    file_path = DATA_DIR / f"{table_name}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Table {table_name} not found at {file_path}")
    
    return file_path.read_text()


def get_all_csv_as_string() -> str:
    """Get all CSV files concatenated as a string with table headers."""
    result = []
    for csv_file in sorted(DATA_DIR.glob("*.csv")):
        table_name = csv_file.stem
        content = csv_file.read_text()
        result.append(f"=== TABLE: {table_name} ===\n{content}")
    return "\n\n".join(result)


def get_all_data_as_json() -> str:
    """Get all tables as JSON format."""
    tables = load_all_tables()
    
    data = {}
    for table_name, df in sorted(tables.items()):
        data[table_name] = df.to_dict(orient="records")
    
    return json.dumps(data, indent=2, default=str)

