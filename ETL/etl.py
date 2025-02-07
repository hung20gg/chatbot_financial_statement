import sys 
import os
import pandas as pd
import numpy as np

current_path = os.path.dirname(__file__)
sys.path.append(current_path)

from crawler.cafef_crawler import CafeFCrawlerFS
from ratio_index import calculate_index

class ETL_Args:
    non_bank_stock_codes = ["MWG", "NHA", "VNM", "HPG", "VHM", "PNJ", "YEG", "FPT","MSN", "GAS", "VRE", "VJC", "VIC", "PLX", "SAB", "POW", "GVR", "BCM", "VPI", "DVM", "KDH", "HDC", "TCH", "CEO", "HUT", "NVL", "DBC", "SAF", "DHT", "VTP", "PVT", "FRT", "DGC", "DCM", "NKG", "CMG", "VGI", "PVC", "CAP", "DTD", "HLD", "L14", "L18", "LAS", "LHC", "NTP", "PLC", "PSD", "PVG", "PVS", "SLS", "TIG", "TMB", "TNG", "TVD", "VC3", "VCS", "DXG"]
    bank_stock_codes = ["BID", "EIB", "OCB", "CTG", "VCB", "ACB", "MBB", "HDB", "TPB", "VPB",  "STB", "TCB",  "SHB", "VIB", "CTG",  "ABB", "LPB", "NVB"]
    securities_stock_codes = ["MBS", "VND", "SSI", "VIX", "ORS"]
    
class SimpleETL:
    def __init__(self, args):
        self.args = args
        self.fs_crawler = CafeFCrawlerFS(args)




def merge_financial_statement(version: str, output_path: str = '../data'):
    
    # Read Mapping file
    df_mapping = pd.read_csv(os.path.join(current_path, f'../csv/{version}/map_category_code_universal.csv')) 

    # Read financial statement
    df_sec = pd.read_parquet(os.path.join(current_path, f'../csv/{version}/securities_financial_report.parquet'))
    df_bank = pd.read_parquet(os.path.join(current_path, f'../csv/{version}/bank_financial_report.parquet'))
    df_corp = pd.read_parquet(os.path.join(current_path, f'../csv/{version}/corp_financial_report.parquet'))
    
    df_bank.rename(columns={'category_code': 'bank_code'}, inplace=True)
    df_sec.rename(columns={'category_code': 'sec_code'}, inplace=True)
    df_corp.rename(columns={'category_code': 'corp_code'}, inplace=True)

    df_sec = pd.merge(df_sec, df_mapping[['sec_code', 'category_code']], how='left', on='sec_code')
    df_corp = pd.merge(df_corp, df_mapping[['corp_code', 'category_code']], how='left', on='corp_code')
    df_bank = pd.merge(df_bank, df_mapping[['bank_code', 'category_code']], how='left', on='bank_code')

    df_bank.drop(columns=['bank_code'], inplace=True)
    df_sec.drop(columns=['sec_code'], inplace=True)
    df_corp.drop(columns=['corp_code'], inplace=True)

    df_fs = pd.concat([df_bank, df_sec, df_corp], ignore_index=True)
    df_fs.dropna(subset=['category_code'], inplace=True)

    df_fs.to_parquet(os.path.join(current_path, output_path, f'financial_statement_{version}.parquet'))

def merge_financial_explaination(output_path: str = '../data'):
    df_sec_tm = pd.read_parquet(os.path.join(current_path,  '../csv/v3/securities_explaination.parquet'))
    df_bank_tm = pd.read_parquet(os.path.join(current_path,  '../csv/v3/bank_explaination.parquet'))
    df_corp_tm = pd.read_parquet(os.path.join(current_path,  '../csv/v3/corp_explaination.parquet'))

    df_tm = pd.concat([df_bank_tm, df_sec_tm, df_corp_tm], ignore_index=True)

    df_tm.to_parquet(os.path.join(current_path, output_path, f'financial_statement_explaination_v3.parquet'))



def calculate_industry_financial_statement(version: str, output_path: str = '../data'):
    company_table = pd.read_csv(os.path.join(current_path, '../csv/df_company_info.csv'))
    df_fs = pd.read_parquet(os.path.join(current_path, output_path, f'financial_statement_{version}.parquet'))
    df_fs = pd.merge(df_fs, company_table[['stock_code', 'industry']], on='stock_code', how='left')

    df_industry_fs = df_fs.groupby(['industry', 'year', 'quarter', 'category_code', 'date_added'])['data'].agg(['sum', 'mean']).reset_index()
    df_industry_fs.rename(columns={'sum': 'data_sum', 'mean': 'data_mean'}, inplace=True)

    df_industry_fs.to_parquet(os.path.join(current_path, output_path, f'industry_report_{version}.parquet'))


def calculate_industry_financial_statement_explaination(output_path: str = '../data'):
    df_tm = pd.read_parquet(os.path.join(current_path, output_path, 'financial_statement_explaination_v3.parquet'))
    company_table = pd.read_csv(os.path.join(current_path, '../csv/df_company_info.csv'))
    df_tm = pd.merge(df_tm, company_table[['stock_code', 'industry']], on='stock_code', how='left')

    df_industry_tm = df_tm.groupby(['industry', 'year', 'quarter', 'category_code', 'date_added'])['data'].agg(['sum', 'mean']).reset_index()
    df_industry_tm.rename(columns={'sum': 'data_sum', 'mean': 'data_mean'}, inplace=True)

    df_industry_tm.to_parquet(os.path.join(current_path, output_path, 'industry_report_explaination_v3.parquet'))


def expand_data(version: str, output_path: str = '../data'):
    
    # Merge financial statement
    merge_financial_statement(version, output_path)

    # Ratio index
    calculate_index(version, output_path)

    # Industry report
    calculate_industry_financial_statement(version, output_path)

    if version == 'v3':
        # Merge financial explan data
        merge_financial_explaination(output_path)

        # Industry report explaination
        calculate_industry_financial_statement_explaination(output_path)

if __name__ == '__main__':
    version = 'v3'
    expand_data(version, '../data')