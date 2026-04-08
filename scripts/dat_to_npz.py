"""
dat_to_npz.py

This script converts .dat files containing SANS (Small Angle Neutron Scattering) detector data 
into .npz files for easier handling and analysis in Python. The script supports processing 
individual files, directories, wildcard patterns, or comma-separated lists of files.

Data such as sample field and temperature are extracted from the metadata comments in the .dat files and
appended to the end of file names. The appended info is the measured temp and field, not the set temp and
field (ie, if you set the temp to 2.5K the metadata value might be 2.6K or something.)

The .dat files are expected to contain metadata in the form of comments (lines starting with '#') 
and numerical data in tabular format. Metadata is parsed and stored in the output .npz file 
alongside the numerical data.

Features:
- Parses metadata and comments from .dat files.
- Converts tabular data into NumPy arrays.
- Supports batch processing of multiple files.
- Saves output in .npz format with metadata and data arrays.

Usage:
    python dat_to_npz.py <input> [-o <output_dir>]

Arguments:
- <input>: Path(s) to .dat file(s), directory, wildcard pattern, or comma-separated list of files.
- -o, --output-dir: Directory where the .npz files will be saved (default: current working directory).

Example:
    python dat_to_npz.py data/*.dat -o output/

"""


import numpy as np
import argparse
from pathlib import Path
import sys
import re



# 5.3 x 4.1 mm pixel shape
def parse_dat(filepath):
    data_rows = []
    metadata = {}
    comments = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    title_set = False

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        if line.startswith("#"):
            meta_line = line.lstrip("#").strip()

            # First metadata line → title
            if not title_set:
                metadata["title"] = meta_line
                title_set = True

                # Extract current from title if present
                match = re.search(r"I=([-\d\.]+)\s*A", meta_line)
                if match:
                    metadata["current"] = float(match.group(1))
                else:
                    metadata["current"] = None

                i += 1
                continue

            # Try structured key-value parsing
            keys = meta_line.split()

            if i + 1 < len(lines) and lines[i + 1].startswith("#"):
                next_line = lines[i + 1].lstrip("#").strip()
                values_str = next_line.split()

                try:
                    values = [float(v) for v in values_str]
                    if len(keys) == len(values):
                        for k, v in zip(keys, values):
                            metadata[k] = v
                        i += 2
                        continue
                except ValueError:
                    pass

            comments.append(meta_line)

        else:
            values = [float(v) for v in line.split()]
            data_rows.append(values)

        i += 1

    data = np.array(data_rows).T
    metadata["comments"] = comments

    return data, metadata


def convert_dat_to_npz(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, metadata = parse_dat(input_path)

    # Extract and round values
    field = metadata.get("sample_field", None)
    temp = metadata.get("sample_temp", None)
    current = metadata.get("current", None)

    if field is not None and temp is not None:
        field_str = f"{np.abs(field):.1f}"
        temp_str = f"{temp:.1f}"
        current_str = f"{current:.3f}A" if current is not None else "none"
        suffix = f"_{field_str}T_{temp_str}K_{current_str}"
    else:
        suffix = "_none"

    output_path = output_dir / (input_path.stem + suffix + ".npz")

    comments = np.array(metadata.get("comments", []))
    np.savez(
        output_path,
        data=data,
        metadata=np.array(metadata, dtype=object)
    )
    print(f"Saved {data.shape} array to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert .dat SANS detector file(s) to .npz"
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Path(s) to input .dat file, directory, wildcard, or comma-separated list of files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=Path.cwd(),
        help="Output directory (default: current working directory)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not args.input:
        print(f"No valid files found for input: {args.input}", file=sys.stderr)
        sys.exit(1)

    for file in args.input:
        convert_dat_to_npz(file, output_dir)


if __name__ == "__main__":
    main()
