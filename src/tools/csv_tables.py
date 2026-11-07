"""Read matrix CSVs without losing STRING identifiers or hiding duplicate headers."""

import csv
from pathlib import Path

import pandas as pd


def read_matrix_csv(path: str | Path, *, strings: bool = False) -> pd.DataFrame:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or len(rows[0]) < 2:
        raise ValueError("CSV must contain an identifier column and value columns")
    header = [name.strip() for name in rows[0]]
    if len(header) != len(set(header)) or any(not name for name in header[1:]):
        raise ValueError("CSV column names must be unique and non-empty")
    records = [row for row in rows[1:] if row]
    if not records or any(len(row) != len(header) for row in records):
        raise ValueError("CSV must contain rows matching its header")
    ids = [row[0].strip() for row in records]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("CSV instance IDs must be unique and non-empty")
    frame = pd.DataFrame([row[1:] for row in records], columns=header[1:], index=ids)
    frame.index.name = header[0] or None
    if not strings:
        frame = frame.apply(pd.to_numeric, errors="raise")
    return frame
