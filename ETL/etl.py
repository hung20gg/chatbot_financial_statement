import sys 
import os
import pandas as pd
import numpy as np

current_path = os.path.dirname(__file__)
sys.path.append(current_path)

# from crawler.cafef_crawler import CafeFCrawlerFS
from ratio_index import calculate_index

# class ETL_Args:
#     non_bank_stock_codes = ["MWG", "NHA", "VNM", "HPG", "VHM", "PNJ", "YEG", "FPT","MSN", "GAS", "VRE", "VJC", "VIC", "PLX", "SAB", "POW", "GVR", "BCM", "VPI", "DVM", "KDH", "HDC", "TCH", "CEO", "HUT", "NVL", "DBC", "SAF", "DHT", "VTP", "PVT", "FRT", "DGC", "DCM", "NKG", "CMG", "VGI", "PVC", "CAP", "DTD", "HLD", "L14", "L18", "LAS", "LHC", "NTP", "PLC", "PSD", "PVG", "PVS", "SLS", "TIG", "TMB", "TNG", "TVD", "VC3", "VCS", "DXG"]
#     bank_stock_codes = ["BID", "EIB", "OCB", "CTG", "VCB", "ACB", "MBB", "HDB", "TPB", "VPB",  "STB", "TCB",  "SHB", "VIB", "CTG",  "ABB", "LPB", "NVB"]
#     securities_stock_codes = ["MBS", "VND", "SSI", "VIX", "ORS"]
    
# class SimpleETL:
#     def __init__(self, args):
#         self.args = args
#         self.fs_crawler = CafeFCrawlerFS(args)




def merge_financial_statement(version: str, output_path: str = '../data'):
    
    # Read Mapping file
    df_mapping = pd.read_csv(os.path.join(current_path, f'../data/{version}/map_category_code_universal.csv')) 

    # Read financial statement
    df_sec = pd.read_parquet(os.path.join(current_path, f'../data/{version}/securities_financial_report.parquet'))
    df_bank = pd.read_parquet(os.path.join(current_path, f'../data/{version}/bank_financial_report.parquet'))
    df_corp = pd.read_parquet(os.path.join(current_path, f'../data/{version}/corp_financial_report.parquet'))
    
    df_bank.rename(columns={'category_code': 'bank_code'}, inplace=True)
    df_sec.rename(columns={'category_code': 'sec_code'}, inplace=True)
    df_corp.rename(columns={'category_code': 'corp_code'}, inplace=True)

    df_sec = pd.merge(df_sec, df_mapping[['sec_code', 'category_code']], how='left', on='sec_code')
    df_corp = pd.merge(df_corp, df_mapping[['corp_code', 'category_code']], how='outer', on='corp_code')
    df_bank = pd.merge(df_bank, df_mapping[['bank_code', 'category_code']], how='left', on='bank_code')

    df_bank.drop(columns=['bank_code'], inplace=True)
    df_sec.drop(columns=['sec_code'], inplace=True)
    df_corp.drop(columns=['corp_code'], inplace=True)

    df_fs = pd.concat([df_bank, df_sec, df_corp], ignore_index=True)
    df_fs.dropna(subset=['stock_code'], inplace=True)

    df_fs.to_parquet(os.path.join(current_path, output_path, f'financial_statement_{version}.parquet'))

def merge_financial_explaination( output_path: str = '../data'):
    df_sec_tm = pd.read_parquet(os.path.join(current_path,  '../data/v3/securities_explaination.parquet'))
    df_bank_tm = pd.read_parquet(os.path.join(current_path,  '../data/v3/bank_explaination.parquet'))
    df_corp_tm = pd.read_parquet(os.path.join(current_path,  '../data/v3/corp_explaination.parquet'))

    df_tm = pd.concat([df_bank_tm, df_sec_tm, df_corp_tm], ignore_index=True)

    df_tm.to_parquet(os.path.join(current_path, output_path, f'financial_statement_explaination_v3.parquet'))



def calculate_industry_financial_statement(version: str, output_path: str = '../data'):
    company_table = pd.read_csv(os.path.join(current_path, '../data/df_company_info.csv'))
    df_fs = pd.read_parquet(os.path.join(current_path, output_path, f'financial_statement_{version}.parquet'))
    df_fs = pd.merge(df_fs, company_table[['stock_code', 'industry']], on='stock_code', how='left')

    df_industry_fs = df_fs.groupby(['industry', 'year', 'quarter', 'category_code', 'date_added'])['data'].agg(['sum', 'mean']).reset_index()
    df_industry_fs.rename(columns={'sum': 'data_sum', 'mean': 'data_mean'}, inplace=True)

    df_industry_fs.to_parquet(os.path.join(current_path, output_path, f'industry_report_{version}.parquet'))


def calculate_industry_financial_statement_explaination(output_path: str = '../data'):
    df_tm = pd.read_parquet(os.path.join(current_path, output_path, 'financial_statement_explaination_v3.parquet'))
    company_table = pd.read_csv(os.path.join(current_path, '../data/df_company_info.csv'))
    df_tm = pd.merge(df_tm, company_table[['stock_code', 'industry']], on='stock_code', how='left')

    df_industry_tm = df_tm.groupby(['industry', 'year', 'quarter', 'category_code', 'date_added'])['data'].agg(['sum', 'mean']).reset_index()
    df_industry_tm.rename(columns={'sum': 'data_sum', 'mean': 'data_mean'}, inplace=True)

    df_industry_tm.to_parquet(os.path.join(current_path, output_path, 'industry_report_explaination_v3.parquet'))


def prepare_files(version: str, extended = False, output_path: str = '../data'):

    if not os.path.exists(os.path.join(current_path, output_path, version)):
        os.makedirs(os.path.join(current_path, output_path, version))

    # Read all the file

    df_company_info = pd.read_csv(os.path.join(current_path, '../csv/df_company_info.csv'))
    df_sub_and_shareholders = pd.read_csv(os.path.join(current_path, '../csv/df_sub_and_shareholders.csv'))
    df_map_ratio_code = pd.read_csv(os.path.join(current_path, f'../csv/map_ratio_code.csv'))

    bank_explaination = pd.read_parquet(os.path.join(current_path, f'../csv/{version}/bank_explaination.parquet'))
    corp_explaination = pd.read_parquet(os.path.join(current_path, f'../csv/{version}/corp_explaination.parquet'))
    securities_explaination = pd.read_parquet(os.path.join(current_path, f'../csv/{version}/securities_explaination.parquet'))

    map_category_code_bank = pd.read_csv(os.path.join(current_path, f'../csv/{version}/map_category_code_bank.csv'))
    map_category_code_corp = pd.read_csv(os.path.join(current_path, f'../csv/{version}/map_category_code_corp.csv'))
    map_category_code_securities = pd.read_csv(os.path.join(current_path, f'../csv/{version}/map_category_code_sec.csv'))
    map_category_code_universal = pd.read_csv(os.path.join(current_path, f'../csv/{version}/map_category_code_universal.csv'))

    if version == 'v3':
        map_category_code_explaination = pd.read_csv(os.path.join(current_path, f'../csv/{version}/map_category_code_explaination.csv'))
        bank_financial_report = pd.read_parquet(os.path.join(current_path, f'../csv/{version}/bank_financial_report.parquet'))
        corp_financial_report = pd.read_parquet(os.path.join(current_path, f'../csv/{version}/corp_financial_report.parquet'))
        securities_financial_report = pd.read_parquet(os.path.join(current_path, f'../csv/{version}/securities_financial_report.parquet'))

    # Merge data if extended
    if extended:
        df_company_info = pd.read_csv(os.path.join(current_path, '../csv/new/df_company_info.csv'))
        df_sub_and_shareholders = pd.read_csv(os.path.join(current_path, '../csv/new/df_sub_and_shareholders.csv'))

        print('===== Extended to',len(df_company_info) ,'companies ======')

        # Need code to change map code here
        # @pphanhh


        
        bank_financial_report2 = pd.read_parquet(os.path.join(current_path, f'../csv/new/{version}/bank_financial_report.parquet'))
        corp_financial_report2 = pd.read_parquet(os.path.join(current_path, f'../csv/new/{version}/corp_financial_report.parquet'))
        securities_financial_report2 = pd.read_parquet(os.path.join(current_path, f'../csv/new/{version}/securities_financial_report.parquet'))

        bank_financial_report = pd.concat([bank_financial_report, bank_financial_report2], ignore_index=True)
        corp_financial_report = pd.concat([corp_financial_report, corp_financial_report2], ignore_index=True)
        securities_financial_report = pd.concat([securities_financial_report, securities_financial_report2], ignore_index=True)

        if version == 'v3':
            

            bank_explaination2 = pd.read_parquet(os.path.join(current_path, f'../csv/new/{version}/bank_explaination.parquet'))
            corp_explaination2 = pd.read_parquet(os.path.join(current_path, f'../csv/new/{version}/corp_explaination.parquet'))
            securities_explaination2 = pd.read_parquet(os.path.join(current_path, f'../csv/new/{version}/securities_explaination.parquet'))


            bank_explaination = pd.concat([bank_explaination, bank_explaination2], ignore_index=True)
            corp_explaination = pd.concat([corp_explaination, corp_explaination2], ignore_index=True)
            securities_explaination = pd.concat([securities_explaination, securities_explaination2], ignore_index=True)



    
    print('===== Using',len(df_company_info) ,'companies ======')
    bank_unique_stock_code = bank_explaination['stock_code'].nunique()
    corp_unique_stock_code = corp_explaination['stock_code'].nunique()
    securities_unique_stock_code = securities_explaination['stock_code'].nunique()

    print('===== Bank:', bank_unique_stock_code, 'Corp:', corp_unique_stock_code, 'Securities:', securities_unique_stock_code, '=====')

    assert len(df_company_info) == bank_unique_stock_code + corp_unique_stock_code + securities_unique_stock_code + 4, 'Number of companies is not correct'

    
    # Save all the file into the output path
    
    df_company_info.to_csv(os.path.join(current_path, output_path, 'df_company_info.csv'), index=False)
    df_sub_and_shareholders.to_csv(os.path.join(current_path, output_path, 'df_sub_and_shareholders.csv'), index=False)
    df_map_ratio_code.to_csv(os.path.join(current_path, output_path, 'map_ratio_code.csv'), index=False)

    bank_explaination.to_parquet(os.path.join(current_path, output_path, version, f'bank_explaination.parquet'))
    corp_explaination.to_parquet(os.path.join(current_path, output_path, version, f'corp_explaination.parquet'))
    securities_explaination.to_parquet(os.path.join(current_path, output_path, version, f'securities_explaination.parquet'))

    map_category_code_bank.to_csv(os.path.join(current_path, output_path, version, 'map_category_code_bank.csv'), index=False)
    map_category_code_corp.to_csv(os.path.join(current_path, output_path, version, 'map_category_code_corp.csv'), index=False)
    map_category_code_securities.to_csv(os.path.join(current_path, output_path, version, 'map_category_code_sec.csv'), index=False)
    map_category_code_universal.to_csv(os.path.join(current_path, output_path, version, 'map_category_code_universal.csv'), index=False)

    if version == 'v3':
        map_category_code_explaination.to_csv(os.path.join(current_path, output_path, version, 'map_category_code_explaination.csv'), index=False)
        bank_financial_report.to_parquet(os.path.join(current_path, output_path, version, f'bank_financial_report.parquet'))
        corp_financial_report.to_parquet(os.path.join(current_path, output_path, version, f'corp_financial_report.parquet'))
        securities_financial_report.to_parquet(os.path.join(current_path, output_path, version, f'securities_financial_report.parquet'))

    return bank_unique_stock_code + corp_unique_stock_code + securities_unique_stock_code


def expand_data(version: str, output_path: str = '../data'):

    versions = version.split('.')
    prefix_version = versions[0]
    suffix_version = '1'
    if len(versions) > 1:
        suffix_version = versions[1]

    expand = False
    if suffix_version == '2':
        expand = True

    num_stock_code = prepare_files(prefix_version, expand, output_path)
    
    # Merge financial statement
    merge_financial_statement(prefix_version, output_path)

    # Ratio index
    df_ratio,_ = calculate_index(prefix_version, output_path)

    ratio_stock_code = df_ratio['stock_code'].nunique()
    print('===== Ratio:', ratio_stock_code, '=====')
    assert num_stock_code == ratio_stock_code, 'Number of companies is not correct'

    # Industry report
    calculate_industry_financial_statement(prefix_version, output_path)

    if 'v3' in version:
        # Merge financial explan data
        merge_financial_explaination(output_path)

        # Industry report explaination
        calculate_industry_financial_statement_explaination(output_path)

if __name__ == '__main__':
    version = 'v3.2'
    expand_data(version, '../data')