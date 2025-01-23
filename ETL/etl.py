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



def calculate_industry_report(version: str, output_path: str = '../data'):
    company_table = pd.read_csv(os.path.join(current_path, '../csv/df_company_info.csv'))
    df_fs = pd.read_parquet(os.path.join(current_path, f'../csv/financial_statement_{version}.parquet'))
    df_fs = pd.merge(df_fs, company_table[['stock_code', 'industry']], on='stock_code', how='left')

    df_industry_fs = df_fs.groupby(['industry', 'year', 'quarter', 'category_code'])['data'].mean().reset_index()
    df_industry_fs.rename(columns={'sum': 'data_sum', 'mean': 'data_mean'}, inplace=True)

    df_industry_fs.to_parquet(os.path.join(current_path, output_path, f'industry_report_{version}.parquet'))


def explan_data(version: str, output_path: str = '../data'):
    
    # Ratio index
    calculate_index(version, output_path)

    # Industry report
    calculate_industry_report(version, output_path)

if __name__ == '__main__':
    version = 'v3'
    explan_data(version, '../data')