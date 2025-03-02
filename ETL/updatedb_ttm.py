import os
import pandas as pd
import numpy as np
import copy 
import const

current_path = os.path.dirname(__file__)  # Path to etl/
project_root = os.path.abspath(os.path.join(current_path, os.pardir))  # Move up one level
data_folder = os.path.join(project_root, "data")  
 

def calculate_TTM(financial_report):
    financial_report = financial_report[financial_report['quarter'] != 0]

    financial_report = financial_report.sort_values(by=['stock_code', 'category_code', 'year', 'quarter'])

    financial_report['period'] = financial_report['year'] * 4 + financial_report['quarter'] - 1

    financial_report['data_TTM'] = (
        financial_report.groupby(['stock_code', 'category_code'])['data']
        .transform(lambda x: x.rolling(window=4, min_periods=4).sum())
    )

    financial_report['date_added_TTM'] = (
        financial_report.groupby(['stock_code', 'category_code'])['date_added']
        .transform(lambda x: x.shift(0))
    )
    financial_report = financial_report.dropna(subset=['data_TTM'])

    financial_report['category_code'] = financial_report['category_code'] + '_TTM'

    return financial_report[['stock_code', 'category_code', 'year', 'quarter', 'data_TTM', 'date_added_TTM']].rename(
        columns={'data_TTM': 'data', 'date_added_TTM': 'date_added'}
    )


def get_unnecessary_code(df_mapping_universal: pd.DataFrame, company_type: str) -> list[str]: # either file path or com
    if 'bank' in company_type:
        df_crop = df_mapping_universal[['bank_code', 'category_code']]
        df_crop = df_crop.rename(columns={'bank_code': 'code'})
    elif 'corp' in company_type:
        df_crop = df_mapping_universal[['corp_code', 'category_code']]
        df_crop = df_crop.rename(columns={'corp_code': 'code'})
    else:
        df_crop = df_mapping_universal[['sec_code', 'category_code']]
        df_crop = df_crop.rename(columns={'sec_code': 'code'})
    
    # Mask drop category_code
    mask_drop_1 = df_crop['category_code'].isin(const.IGNORE_TTM_CODES_IS)
    mask_drop_2 = df_crop['category_code'].isin(const.IGNORE_TTM_CODES_CF)
    df_crop = df_crop[~(mask_drop_1 | mask_drop_2)]
    crop_code = df_crop['code'].unique()

    return crop_code


def process_financial_statements(input_parquet_path: str, output_parquet_path: str,company_type:str = None, version = 'v3') -> pd.DataFrame: 
    combined_report = None

    if company_type:
        df_mapping_universal = pd.read_csv(os.path.join(data_folder, version, "map_category_code_universal.csv"))

        crop_code = get_unnecessary_code(df_mapping_universal, company_type)
    else:
        ignore_code = const.IGNORE_TTM_CODES_IS + const.IGNORE_TTM_CODES_CF
        all_code = pd.read_csv(os.path.join(data_folder, version, "map_category_code_universal.csv"))['category_code'].unique()
        crop_code = [code for code in all_code if code not in ignore_code]


    if os.path.exists(input_parquet_path):

        financial_report = pd.read_parquet(input_parquet_path)

        financial_report = financial_report.dropna(subset=['category_code'])
        print(f"Processing {company_type or 'Universal'} Financial Report")

        is_report = financial_report[financial_report['category_code'].str.startswith('IS')]
        cf_report = financial_report[financial_report['category_code'].str.startswith('CF')]

        is_report = is_report[is_report['category_code'].isin(crop_code)]
        cf_report = cf_report[cf_report['category_code'].isin(crop_code)]

        is_report_TTM = calculate_TTM(is_report.copy()) 
        cf_report_TTM = calculate_TTM(cf_report.copy()) 

        combined_report = pd.concat([financial_report, is_report_TTM, cf_report_TTM], ignore_index=True)
        combined_report.to_parquet(output_parquet_path, index=False)
        print(f"Financial statement saved to {output_parquet_path}")

    else:
        print(f"File not found: {input_parquet_path}. Skipping.")

    return combined_report
    

def process_map_universal(input_csv_path: str, output_csv_path: str, version = 'v3') -> pd.DataFrame: 
    
    concate_df = None
    if os.path.exists(input_csv_path):
        print(f"Processing: {input_csv_path}")
        map_universal = pd.read_csv(input_csv_path)

        print("Existing columns:", list(map_universal.columns))

        filename = os.path.basename(input_csv_path)

        df_map_universal = pd.read_csv(os.path.join(data_folder, version, "map_category_code_universal.csv"))

        if filename == "map_category_code_universal.csv":

            filtered_df = map_universal[map_universal['category_code'].str.startswith(('IS_', 'CF_'), na=False)].copy()

            remove_mask_is = filtered_df['category_code'].isin(const.IGNORE_TTM_CODES_IS)
            remove_mask_cf = filtered_df['category_code'].isin(const.IGNORE_TTM_CODES_CF)

            filtered_df = filtered_df[~(remove_mask_is | remove_mask_cf)]

            filtered_df['category_code'] = filtered_df['category_code'] + '_TTM'

            print("Adding TTM suffix to category_code", filtered_df['category_code'].nunique())

            for col in ['corp_code', 'sec_code', 'bank_code']:
                if col in filtered_df.columns:
                    # if filtered_df[col]:
                        filtered_df[col] = filtered_df[col].astype(str) + '_TTM'

            for col in ['Corp', 'Securities', 'Bank', 'en_caption']:
                if col in filtered_df.columns:
                    # if filtered_df[col]:
                        filtered_df[col] = filtered_df[col].astype(str) + ' (Trailing Twelve Months)'

            # Handle NaN cases (if some columns had missing values)
            filtered_df.replace({'nan (Trailing Twelve Months)': None}, inplace=True)
            filtered_df.replace({'nan_TTM': None}, inplace=True)

            concate_df = pd.concat([map_universal, filtered_df], ignore_index=True)
            concate_df.to_csv(output_csv_path, index=False)
  
            print(f"Universal mapping saved to {output_csv_path}")

        
        
        elif filename in ["map_category_code_bank.csv", "map_category_code_corp.csv", "map_category_code_sec.csv"]:
            print(f"Processing {filename} Mapping File")  

            crop_code = get_unnecessary_code(df_map_universal, filename)

            filtered_df = map_universal[map_universal['category_code'].str.startswith(('IS_', 'CF_'), na=False)].copy()

            filtered_df = filtered_df[filtered_df['category_code'].isin(crop_code)]

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
