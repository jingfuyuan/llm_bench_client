from pathlib import Path
import pandas as pd
from glob import glob
import argparse
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description="Collect CSV data from specified directory.")
    parser.add_argument('--input_dir', required=True, help='List of input directories containing CSV files.')
    parser.add_argument('--output_file', required=True, help='Output CSV file to save the collected data.')
    return parser.parse_args()

def collect_data(input_dir, output_file):
    # find all csv files that have a name pattern *_summary.csv in the input directory and its subdirectories
    input_dir = Path(input_dir)
    csv_files = glob(str(input_dir / '**' / '*_summary.csv'), recursive=True)
    assert len(csv_files) > 0, f"No CSV files found in {input_dir} matching pattern '*_summary.csv'"
    csv_files = [Path(f) for f in sorted(csv_files)]
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df["file_path"] = str(csv_file.absolute())
        # extract batch size and repeat number from the file name assuming the file name pattern is like '*-bs{batch_size}-rep{repeat_number}_summary.csv'
        file_name = csv_file.name
        df["directory"] = str(csv_file.parent)
        df["file_name"] = file_name
        # extract batch size using regex
        bs_match = re.search(r'-bs(\d+)', file_name)
        df['batch_size'] = int(bs_match.group(1)) if bs_match else None
        # extract repeat number using regex
        rep_match = re.search(r'-rep(\d+)', file_name)
        df['repeat_number'] = int(rep_match.group(1)) if rep_match else None

        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    # change the order of columns to have 'file_path' as the first column
    cols = combined_df.columns.tolist()
    cols = cols[-5:] + cols[:-5]  # move last 5 columns to the front
    combined_df = combined_df[cols]
    combined_df.to_csv(output_file, index=False)
    # print a message indicating how many files have been processed and where the data has been saved
    print(f"Processed {len(csv_files)} files. Collected data saved to {output_file}.")


if __name__ == "__main__":
    args = parse_arguments()
    collect_data(args.input_dir, args.output_file)
