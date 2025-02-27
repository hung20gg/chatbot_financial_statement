import os
import pandas as pd
import numpy as np
import copy 

current_path = os.path.dirname(__file__)  # Path to etl/
project_root = os.path.abspath(os.path.join(current_path, os.pardir))  # Move up one level
data_folder = os.path.join(project_root, "data")  
 

def process_financial_statements(input_parquet_path: str, output_parquet_path: str,company_type:str = None) -> pd.DataFrame: 
    """
    Processes financial statements and computes TTM rolling sums.
    """
    combined_report = None
    if os.path.exists(input_parquet_path):

        financial_report = pd.read_parquet(input_parquet_path)
        financial_report = financial_report.dropna(subset=['category_code'])

        def calculate_TTM(df):
            df_sorted = df.sort_values(by=['stock_code', 'category_code', 'year', 'quarter']).reset_index(drop=True)
            df_sorted['period'] = df_sorted['year'] * 4 + df_sorted['quarter'] - 1  

            df_lookup = df_sorted[['stock_code', 'category_code', 'year', 'quarter', 'data', 'period']].copy()
            df_lookup['lookback1'] = df_lookup['period']
            df_lookup['lookback2'] = df_lookup['period'] - 1
            df_lookup['lookback3'] = df_lookup['period'] - 2
            df_lookup['lookback4'] = df_lookup['period'] - 3

            df_long = df_lookup.melt(
                id_vars=['stock_code', 'category_code', 'data', 'period'],
                value_vars=['lookback1', 'lookback2', 'lookback3', 'lookback4'],
                var_name='lookback_type',
                value_name='lookup_period'
            )

            df_merged = df_long.merge(
                df_sorted[['stock_code', 'category_code', 'period', 'data']],
                left_on=['stock_code', 'category_code', 'lookup_period'],
                right_on=['stock_code', 'category_code', 'period'],
                how='left',
                suffixes=('', '_past')
            )

            df_final = df_merged.groupby(['stock_code', 'category_code', 'period'])['data_past'].sum().reset_index()
            df_final['year'] = df_final['period'] // 4
            df_final['quarter'] = df_final['period'] % 4 + 1
            df_final.rename(columns={'data_past': 'data'}, inplace=True)

            return df_final

        print(f"Processing {company_type or 'Universal'} Financial Report")

        is_report = financial_report[financial_report['category_code'].str.startswith('IS')]
        cf_report = financial_report[financial_report['category_code'].str.startswith('CF')]

        is_report_TTM = calculate_TTM(is_report.copy()) if not is_report.empty else None
        cf_report_TTM = calculate_TTM(cf_report.copy()) if not cf_report.empty else None

        report_TTM = pd.concat([df for df in [is_report_TTM, cf_report_TTM] if df is not None], ignore_index=True)

        report_TTM['category_code'] = report_TTM['category_code'] + '_TTM'
        report_TTM['original_category_code'] = report_TTM['category_code'].str.replace('_TTM', '', regex=True)

        report_TTM = report_TTM.merge(
            financial_report[['stock_code', 'year', 'quarter', 'category_code', 'date_added']],
            left_on=['stock_code', 'year', 'quarter', 'original_category_code'],
            right_on=['stock_code', 'year', 'quarter', 'category_code'],
            how='left',
            suffixes=('', '_original')
        )
        report_TTM['date_added'] = report_TTM['date_added'].combine_first(financial_report.get('date_added'))
        report_TTM.drop(columns=['category_code_original', 'original_category_code'], inplace=True)

        missing_cols = [col for col in financial_report.columns if col not in report_TTM.columns]
        for col in missing_cols:
            report_TTM[col] = None

        report_TTM = report_TTM[financial_report.columns]

        combined_report = pd.concat([financial_report, report_TTM], ignore_index=True)
    
        combined_report.to_parquet(output_parquet_path, index=False)
        print(f"Financial statement saved to {output_parquet_path}")

    else:
        print(f"File not found: {input_parquet_path}. Skipping.")

    return combined_report

def process_map_universal(input_csv_path: str, output_csv_path: str) -> pd.DataFrame: 
    
    concate_df = None
    if os.path.exists(input_csv_path):
        print(f"Processing: {input_csv_path}")
        map_universal = pd.read_csv(input_csv_path)

        print("Existing columns:", list(map_universal.columns))

        filename = os.path.basename(input_csv_path)

        if filename == "map_category_code_universal.csv":

            filtered_df = map_universal[map_universal['category_code'].str.startswith(('IS_', 'CF_'), na=False)].copy()

            filtered_df['category_code'] = filtered_df['category_code'] + '_TTM'

            for col in ['corp_code', 'sec_code', 'bank_code']:
                if col in filtered_df.columns:
                    filtered_df[col] = filtered_df[col].astype(str) + '_TTM'

            for col in ['Corp', 'Securities', 'Bank', 'en_caption']:
                if col in filtered_df.columns:
                    filtered_df[col] = filtered_df[col].astype(str) + ' (Trailing Twelve Months)'

            # Handle NaN cases (if some columns had missing values)
            filtered_df.replace({'nan (Trailing Twelve Months)': None}, inplace=True)
            filtered_df.replace({'nan_TTM': None}, inplace=True)

            concate_df = pd.concat([map_universal, filtered_df], ignore_index=True)
            concate_df.to_csv(output_csv_path, index=False)
  
            print(f"Universal mapping saved to {output_csv_path}")

        elif filename in ["map_category_code_bank.csv", "map_category_code_corp.csv", "map_category_code_sec.csv"]:
            print(f"Processing {filename} Mapping File")  

            filtered_df = map_universal[map_universal['category_code'].str.startswith(('IS_', 'CF_'), na=False)].copy()

            filtered_df['category_code'] = filtered_df['category_code'] + '_TTM'

            for col in ['old_caption', 'vi_caption', 'en_caption']:
                if col in filtered_df.columns:
                    filtered_df[col] = filtered_df[col].astype(str) + ' (Trailing Twelve Months)'

            filtered_df.replace({'nan (Trailing Twelve Months)': None}, inplace=True)
            filtered_df.replace({'nan_TTM': None}, inplace=True)

            concate_df = pd.concat([map_universal, filtered_df], ignore_index=True)
            concate_df.to_csv(output_csv_path, index=False)
       
            print(f"{filename} mapping saved to {output_csv_path}")

        else:
            print(f"Unrecognized file: {input_csv_path}. Skipping.")

    else:
        print(f"File not found: {input_csv_path}. Skipping.")

    return concate_df


if __name__ == "__main__":
    process_financial_statements(
        os.path.join(data_folder, "financial_statement_v3.parquet"),
        os.path.join(data_folder, "financial_statement_v3.parquet")
    )

    for company_type in ["bank", "corp", "securities"]:
        process_financial_statements(
            os.path.join(data_folder,'v3', f"{company_type}_financial_report.parquet"),
            os.path.join(data_folder,'v3', f"{company_type}_financial_report.parquet"),
            company_type = company_type
        )

    mapping_files = [
        "map_category_code_universal.csv",
        "map_category_code_bank.csv",
        "map_category_code_corp.csv",
        "map_category_code_sec.csv"
    ]

    for mapping_file in mapping_files:
        input_csv = os.path.join(data_folder,'v3', mapping_file)
        output_csv = os.path.join(data_folder,'v3', mapping_file)
        process_map_universal(input_csv, output_csv)
