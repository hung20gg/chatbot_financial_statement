import pandas as pd
import os

# Ensure directories exist
os.makedirs("../data", exist_ok=True)
os.makedirs("../data", exist_ok=True)

def process_financial_statements(input_parquet_path, output_parquet_path):
    """
    Processes financial statements by computing 4NQ rolling sums and merging with universal report.
    Saves the final combined report to a Parquet file.
    """

    # Load the universal report
    universal_report = pd.read_parquet(input_parquet_path)

    def calculate_4NQ(df):

        df_sorted = df.sort_values(by=['stock_code', 'category_code', 'year', 'quarter']).reset_index(drop=True)
        df_sorted['period'] = df_sorted['year'] * 4 + df_sorted['quarter'] - 1 # year * 4 + quarter - 1

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

    is_report = universal_report[universal_report['category_code'].str.startswith('IS')]
    cf_report = universal_report[universal_report['category_code'].str.startswith('CF')]

    is_report = is_report[is_report['quarter'] != 0]
    cf_report = cf_report[cf_report['quarter'] != 0]

    is_report_4NQ = calculate_4NQ(is_report)
    cf_report_4NQ = calculate_4NQ(cf_report)

    report_4NQ = pd.concat([is_report_4NQ, cf_report_4NQ], ignore_index=True)

    report_4NQ['category_code'] = report_4NQ['category_code'] + '_4NQ'

    report_4NQ['original_category_code'] = report_4NQ['category_code'].str.replace('_4NQ', '', regex=True)
    report_4NQ = report_4NQ.merge(
        universal_report[['stock_code', 'year', 'quarter', 'category_code', 'date_added']],
        left_on=['stock_code', 'year', 'quarter', 'original_category_code'],
        right_on=['stock_code', 'year', 'quarter', 'category_code'],
        how='left',
        suffixes=('', '_original')
    )
    report_4NQ.drop(columns=['category_code_original', 'original_category_code'], inplace=True)

    missing_cols = [col for col in universal_report.columns if col not in report_4NQ.columns]
    for col in missing_cols:
        report_4NQ[col] = None  

    report_4NQ = report_4NQ[universal_report.columns]

    combined_report = pd.concat([universal_report, report_4NQ], ignore_index=True)

    combined_report.to_parquet(output_parquet_path, index=False)
    print(f"Financial statement saved to {output_parquet_path}")

def process_map_universal(input_csv_path, output_csv_path):
    """
    Processes the map_universal dataset by adding _4NQ records and updating captions.
    Saves the final dataframe to a CSV file.
    """
    map_universal = pd.read_csv(input_csv_path)

    filtered_df = map_universal[map_universal['category_code'].str.startswith(('IS_', 'CF_'), na=False)].copy()
    
    filtered_df['category_code'] = filtered_df['category_code'] + '_4NQ'
    filtered_df['corp_code'] = filtered_df['corp_code'].astype(str) + '_4NQ'
    filtered_df['sec_code'] = filtered_df['sec_code'].astype(str) + '_4NQ'
    filtered_df['bank_code'] = filtered_df['bank_code'].astype(str) + '_4NQ'

    filtered_df['en_caption'] = filtered_df['en_caption'].astype(str) + ' (4 nearest quarters)'
    filtered_df['Corp'] = filtered_df['Corp'].astype(str) + ' (4 nearest quarters)'
    filtered_df['Securities'] = filtered_df['Securities'].astype(str) + ' (4 nearest quarters)'
    filtered_df['Bank'] = filtered_df['Bank'].astype(str) + ' (4 nearest quarters)'

    filtered_df.replace({'nan (4 nearest quarters)': None, 'nan_4NQ': None}, inplace=True)

    map_universal_extended = pd.concat([map_universal, filtered_df], ignore_index=True)

    # Save to CSV
    map_universal_extended.to_csv(output_csv_path, index=False)
    print(f"Map universal saved to {output_csv_path}")

if __name__ == "__main__":
    
    input_parquet_path = r"D:\python\financial statement prj\chatbot_financial_statement\data\financial_statement_v3.parquet"
    output_parquet_path = "../data/financial_statement_v3_2.parquet"

    input_csv_path = r"D:\python\financial statement prj\chatbot_financial_statement\csv\v3\map_category_code_universal.csv"
    output_csv_path = "../data/map_category_code_universal.csv"

    process_financial_statements(input_parquet_path, output_parquet_path)
    process_map_universal(input_csv_path, output_csv_path)
