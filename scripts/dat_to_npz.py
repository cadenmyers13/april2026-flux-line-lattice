import numpy as np
import argparse
from pathlib import Path
import sys


def parse_dat(filepath):
    data_rows = []
    metadata = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                meta_line = line.lstrip("#").strip()
                if meta_line:
                    metadata.setdefault("comments", []).append(meta_line)
            else:
                values = [float(v) for v in line.split()]
                data_rows.append(values)

    data = np.array(data_rows)
    # Transpose to fix axes orientation
    data = data.T
    return data, metadata


def dat_to_npz(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, metadata = parse_dat(input_path)
    output_path = output_dir / (input_path.stem + ".npz")
    comments = np.array(metadata.get("comments", []))
    np.savez(output_path, data=data, comments=comments)
    print(f"Saved {data.shape} array to {output_path}")
    return output_path


def expand_inputs(input_arg):
    """Expand a single input argument into a list of Path objects."""
    paths = []
    input_arg = input_arg.strip()

    # Check for comma-separated list
    if "," in input_arg:
        for p in input_arg.split(","):
            paths.extend(expand_inputs(p))
        return paths

    p = Path(input_arg)

    # Directory: include all .dat files
    if p.is_dir():
        paths.extend(sorted(p.glob("*.dat")))
    # Wildcard pattern: e.g., *.dat
    elif "*" in input_arg or "?" in input_arg:
        paths.extend(sorted(Path().glob(input_arg)))
    # Single file
    elif p.is_file():
        paths.append(p)
    else:
        print(f"Warning: no files found for {input_arg}", file=sys.stderr)

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Convert .dat SANS detector file(s) to .npz"
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Path(s) to input .dat file, directory, wildcard, or comma-separated list of files"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=Path.cwd(),
        help="Output directory (default: current working directory)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    files = []
    for inp in args.input:
        files.extend(expand_inputs(inp))
    if not files:
        print(f"No valid files found for input: {args.input}", file=sys.stderr)
        sys.exit(1)

    for f in files:
        dat_to_npz(f, output_dir)


if __name__ == "__main__":
    main()