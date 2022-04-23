import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path')
    args = parser.parse_args()

    pd.set_option('display.max_rows', None)
    df = pd.read_csv(args.csv_path)
    print(df)
