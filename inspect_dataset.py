
import csv
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent / "backend" / "data" / "raw"


def inspect_csv(file_path: Path) -> None:
    print("\n" + "=" * 65)
    print(f"FILE: {file_path.name}")
    print("=" * 65)

    with file_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            print("ERROR: No CSV header was found.")
            return

        print("Columns:", reader.fieldnames)

        total_rows = 0
        empty_counts = Counter()
        category_counts = Counter()
        seen_rows = set()
        duplicate_rows = 0
        examples = []

        for row in reader:
            total_rows += 1

            # Count empty values in every column.
            for column, value in row.items():
                if value is None or not value.strip():
                    empty_counts[column] += 1

            # Optional category statistics.
            if "Category" in row and row["Category"]:
                category_counts[row["Category"].strip()] += 1

            # Identify identical complete rows.
            row_key = tuple(
                (key, value) for key, value in row.items()
            )
            if row_key in seen_rows:
                duplicate_rows += 1
            else:
                seen_rows.add(row_key)

            if len(examples) < 2:
                examples.append(row)

        print("Total data rows:", total_rows)
        print("Completely duplicated rows:", duplicate_rows)
        print("Empty values by column:", dict(empty_counts))
        print("Category counts:", dict(category_counts))

        print("\nSample records:")
        for index, example in enumerate(examples, start=1):
            print(f"\nRecord {index}:")
            for column, value in example.items():
                # Avoid printing entire article bodies.
                preview = (value or "")[:180]
                print(f"  {column}: {preview!r}")


def main() -> None:
    if not DATA_DIR.exists():
        print(f"Dataset folder not found: {DATA_DIR}")
        return

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {DATA_DIR}")
        print("Extract the downloaded dataset into this folder.")
        return

    print("Dataset folder:", DATA_DIR)
    print("CSV files found:", len(csv_files))

    for file_path in csv_files:
        inspect_csv(file_path)


if __name__ == "__main__":
    main()
