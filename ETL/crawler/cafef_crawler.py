import pandas as pd
import numpy as np
import os 
import re
import logging
import time
import sys 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
current_dir = os.path.dirname(os.path.abspath(__file__))

import requests
from bs4 import BeautifulSoup

from .base import AbstractCrawlerFS


def insert_null_row(df, index_to_insert):
    null_row = pd.DataFrame([[np.nan]*df.shape[1]], columns=df.columns)
    df_with_null = pd.concat([df.iloc[:index_to_insert], null_row, df.iloc[index_to_insert:]]).reset_index(drop=True)
    return df_with_null


# Category code mapping

map_caption_bcdkt_bank = pd.read_csv(os.path.join(current_dir, r'mapping_data/map_caption_bcdkt_bank.csv'))
map_caption_bkqkd_bank = pd.read_csv(os.path.join(current_dir, r'mapping_data/map_caption_bkqkd_bank.csv'))
map_caption_blctt_bank = pd.read_csv(os.path.join(current_dir, r'mapping_data/map_caption_blctt_bank.csv'))

map_caption_bcdkt_non_bank = pd.read_csv(os.path.join(current_dir, r'mapping_data/map_caption_bcdkt_non_bank.csv'))
map_caption_bkqkd_non_bank = pd.read_csv(os.path.join(current_dir, r'mapping_data/map_caption_bkqkd_non_bank.csv'))
map_caption_blctt_non_bank = pd.read_csv(os.path.join(current_dir, r'mapping_data/map_caption_blctt_non_bank.csv'))

map_caption_bcdkt_sec = pd.read_csv(os.path.join(current_dir, r'mapping_data/map_caption_bcdkt_sec.csv'))
map_caption_bkqkd_sec = pd.read_csv(os.path.join(current_dir, r'mapping_data/map_caption_bkqkd_sec.csv'))
map_caption_blctt_sec = pd.read_csv(os.path.join(current_dir, r'mapping_data/map_caption_blctt_sec.csv'))

# Dictionary mapping
dict_map_caption_bcdkt_bank = dict(zip(map_caption_bcdkt_bank['vi_caption'], map_caption_bcdkt_bank['category_code']))
dict_map_caption_bkqkd_bank = dict(zip(map_caption_bkqkd_bank['vi_caption'], map_caption_bkqkd_bank['category_code']))
dict_map_caption_blctt_bank = dict(zip(map_caption_blctt_bank['vi_caption'], map_caption_blctt_bank['category_code']))

dict_map_caption_bcdkt_non_bank = dict(zip(map_caption_bcdkt_non_bank['vi_caption'], map_caption_bcdkt_non_bank['category_code']))
dict_map_caption_bkqkd_non_bank = dict(zip(map_caption_bkqkd_non_bank['vi_caption'], map_caption_bkqkd_non_bank['category_code']))
dict_map_caption_blctt_non_bank = dict(zip(map_caption_blctt_non_bank['vi_caption'], map_caption_blctt_non_bank['category_code']))

dict_map_caption_bcdkt_sec = dict(zip(map_caption_bcdkt_sec['vi_caption'], map_caption_bcdkt_sec['category_code']))
dict_map_caption_bkqkd_sec = dict(zip(map_caption_bkqkd_sec['vi_caption'], map_caption_bkqkd_sec['category_code']))
dict_map_caption_blctt_sec = dict(zip(map_caption_blctt_sec['map_24h_caption'], map_caption_blctt_sec['category_code']))


map_caption_bcdkt_bank2 = insert_null_row(map_caption_bcdkt_bank, 0)
map_caption_bcdkt_bank2 = insert_null_row(map_caption_bcdkt_bank2, 0)
map_caption_bcdkt_bank2 = insert_null_row(map_caption_bcdkt_bank2, 0)


def get_quarter_year(text):
    pattern = r'Quý\s+(\d+)\s*-\s*(\d{4})'
    match = re.search(pattern, text)
    if match:
        quarter = match.group(1)
        year = match.group(2)
        return quarter, year
    else:
        return None, None


def transpose_data(df, stock_code):
    
    stock_codes = []
    categories = []
    years = []
    quarters = []
    data = []
    
    for row in df.iterrows():
        index, data_row = row
        stock_code = stock_code
        for col in df.columns:
            quarter, year = get_quarter_year(col)
            if quarter is None:
                if col.isdigit():
                    quarter = 0
                    year = col
                else:
                    continue
            stock_codes.append(stock_code)
            categories.append(data_row['category_code'])
            years.append(int(year))
            quarters.append(int(quarter))
            data.append(data_row[col])
    return pd.DataFrame({"stock_code": stock_codes, "category_code": categories, "year": years, "quarter": quarters, "data": data})

def map_bank_bsheet_code(df):
    df['category_code'] = map_caption_bcdkt_bank2['category_code']
    df.drop(columns=['category'], inplace=True)
    df.dropna(subset=['category_code'], inplace=True)
    return df

def map_bsheet_code(df, map_df):
    df['category_code'] = map_df['category_code']
    df.drop(columns=['category'], inplace=True)
    df.dropna(subset=['category_code'])
    return df


def map_normal_category_code(df, sheet, is_bank):
    if is_bank:
        if sheet == 'incsta':
            df['category_code'] = df['category'].map(dict_map_caption_bkqkd_bank)
        elif sheet == 'cashflow':
            df['category_code'] = df['category'].map(dict_map_caption_blctt_bank)
    else:
        if sheet == 'incsta':
            df['category_code'] = df['category'].map(dict_map_caption_bkqkd_non_bank)
        elif sheet == 'cashflow':
            df['category_code'] = df['category'].map(dict_map_caption_blctt_non_bank)
    df.drop(columns=['category'], inplace=True)
    df.dropna(subset=['category_code'], inplace=True)
    return df

def map_sec_category_code(df, sheet):
    if sheet == 'bsheet':
        df['category_code'] = df['category'].map(dict_map_caption_bcdkt_sec)
    elif sheet == 'incsta':
        df['category_code'] = df['category'].map(dict_map_caption_bkqkd_sec)
    else:
        df['category_code'] = df['category'].map(dict_map_caption_blctt_sec)
        
    df.drop(columns=['category'], inplace=True)
    df.dropna(subset=['category_code'], inplace=True)
    return df


# Number utils

def chop_number(x):
    x = str(x)
    if len(x) > 16:
        return int(x[:-6])
    return int(x)

def chop_mil(x):
    if abs(x) >=1_000_000:
        return int(x/1_000_000)
    return x

def fix_number_issue(df):
    df['data'] = df['data'].apply(lambda x: str(x).replace(",", ""))
    df['data'] = df['data'].apply(lambda x: chop_number(x))
    df['data'] = df['data'].apply(lambda x: chop_mil(x))
    return df

# Main crawler function

def craw_cafef(stock_code, start_year, start_month, duration = 2, sheet = 'bsheet', is_bank = False):
# Send a GET request to fetch the raw HTML content
    if start_month == 0:
        start_year += 4
    else:
        start_year += 1
    
    dfs = None
    while duration > 0:
        duration -= 1
        if start_month == 0:
            start_year -= 4
        else:
            start_year -= 1
        
        try:
            if sheet == 'bsheet':
                url = f'https://s.cafef.vn/bao-cao-tai-chinh{"-ngan-hang" if is_bank else ""}/{stock_code.lower()}/{sheet}/{start_year}/{start_month}/0/1/bao-cao-tai-chinh-.chn'
            elif sheet == 'incsta':
                url = f'https://s.cafef.vn/bao-cao-tai-chinh{"-ngan-hang" if is_bank else ""}/{stock_code.lower()}/{sheet}/{start_year}/{start_month}/0/0/ket-qua-hoat-dong-kinh-doanh-.chn'
            elif sheet == 'cashflow':
                url = f'https://s.cafef.vn/bao-cao-tai-chinh{"-ngan-hang" if is_bank else ""}/{stock_code.lower()}/{sheet + "direct" if is_bank else sheet}/{start_year}/{start_month}/0/1/luu-chuyen-tien-te-{"truc" if is_bank else "gian"}-tiep-.chn'
            response = requests.get(url)
            html_content = response.text

            # Parse the content with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find the table elements
            tables = soup.find_all('table') # Adjust as necessary depending on the structure

            row = tables[3].find('tr')
            headers = [cell.text.strip() for cell in row.find_all('td')]
            headers[0] = 'category'
            headers.pop()

            rows = tables[4].find_all('tr')
            records = []
            for row in rows:
                record = []
                i = 0
                has_text = False
                for cell in row.find_all('td'):
                    if cell.text.strip() != '':
                        has_text = True
                        record.append(cell.text.strip())
                    else:
                        record.append(np.nan)
                    i+=1
                    if i == len(headers):
                        break
                if len(record) > 0 and has_text:
                    records.append(record)
            df = pd.DataFrame(records, columns=headers)
            # return df
            df = df.dropna(how='all', axis=1)
            
            
        #    Map category code
            if sheet == 'bsheet':
                if is_bank:
                    df = map_bsheet_code(df, map_caption_bcdkt_bank2)
                else:
                    df = map_bsheet_code(df, map_caption_bcdkt_non_bank)
                
            else:
                df = map_normal_category_code(df, sheet, is_bank)
                
            df.dropna(subset=['category_code'], inplace=True)
                
            if dfs is None:
                dfs = df
            else:
                dfs = pd.merge(dfs, df, on='category_code', how= 'left', suffixes=('_df1', '_df2'))
                columns_to_check = df.columns
                for col in columns_to_check:
                    if f"{col}_df1" in dfs.columns:
                        dfs[f"{col}"] = dfs[f"{col}_df1"].fillna(dfs[f"{col}_df2"])
                        dfs.drop(columns=[f"{col}_df1",f"{col}_df2"], inplace=True)
        except:
            print(f"Error at {start_year} - {start_month}")
            continue
               
    if dfs is None:
        return None 
      
    
    dfs = transpose_data(dfs, stock_code.upper())
    dfs.dropna(subset=['category_code'], inplace=True)
    dfs.fillna(0, inplace=True)
    dfs['year'] = dfs['year'].astype(int)
    dfs['quarter'] = dfs['quarter'].astype(int)
    return fix_number_issue(dfs)  


def craw_24h_financial_statement(stock_code, period, view, pages, lang = 'vi'):
    dfs = None
    for page in range(1, pages+1):
        url = f"https://api-finance-t19.24hmoney.vn/v1/ios/company/financial-report?device_name=INVALID&device_model=Windows+11&network_carrier=INVALID&connection_type=INVALID&os=Chrome&os_version=128.0.0.0&access_token=INVALID&push_token=INVALID&locale={lang}&symbol={stock_code}&period={period}&view={view}&page={page}&expanded=true"
        response = requests.get(url)
        
        if response.status_code == 200:
            try:
                data = response.json()
                headers = data['data']['headers']
                rows = data['data']['rows']
                categories = []
                stock_codes = []
                year = []
                quarter = []
                data = []
                
                for i in range(len(rows)):
                    for j in range(0, len(rows[i]),2):
                        
                        stock_codes.append(stock_code)
                        categories.append(rows[i]['name'])
                        year.append(headers[j]['year'])
                        quarter.append(headers[j]['quarter'])   
                        data.append(rows[i]['values'][j])  
                
                df = pd.DataFrame({ 'stock_code': stock_code, 'category': categories, 'year': year, 'quarter': quarter, 'data': data})
                if dfs is None:
                    dfs = df
                else:
                    dfs = pd.concat([dfs, df]).reset_index(drop=True)
                    
                del df
            
            except:
                print(response.json())
                continue    
            
        else:
            print(response.json()) 
    
    return dfs


def crawl_mix_securities(stock_code, start_year, start_month, duration = 2, sheet = 'bsheet'):
    source = 'cafef'
    if sheet == 'cashflow':
        source = 'money24h'
    
    dfs = None   
    
    if source == 'cafef':
        if start_year == 2024 and start_month == 0:
            start_year = 2023
        while duration > 0:
            if sheet == 'bsheet':
                url = f'https://s.cafef.vn/bao-cao-tai-chinh-chung-khoan/{stock_code.lower()}/{sheet}/{start_year}/{start_month}/0/1/bao-cao-tai-chinh-.chn'
            elif sheet == 'incsta':
                url = f'https://s.cafef.vn/bao-cao-tai-chinh-chung-khoan/{stock_code.lower()}/{sheet}/{start_year}/{start_month}/0/0/ket-qua-hoat-dong-kinh-doanh-.chn'
            response = requests.get(url)
            html_content = response.text

            # Parse the content with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            tables = soup.find_all('table') # Adjust as necessary depending on the structure

            row = tables[3].find('tr')
            headers = [cell.text.strip() for cell in row.find_all('td')]
            headers[0] = 'category'
            headers.pop()

            print(headers)

            rows = tables[4].find_all('tr')
            records = []
            for row in rows:
                record = []
                i = 0
                has_text = False
                for cell in row.find_all('td'):
                    if cell.text.strip() != '':
                        has_text = True
                        record.append(cell.text.strip())
                    else:
                        record.append(np.nan)
                    i+=1
                    if i == len(headers):
                        break
                if len(record) > 0 and has_text:
                    records.append(record)
            df = pd.DataFrame(records, columns=headers)
            df = df.dropna(how='all', axis=1)
            
        #    Map category code
            if sheet == 'bsheet':
                df = map_bsheet_code(df, map_caption_bcdkt_sec)
            else:
                df = map_sec_category_code(df, sheet)
            
            df.dropna(subset=['category_code'], inplace=True)
            
            if dfs is None:
                dfs = df
            else:
                dfs = pd.merge(dfs, df, on='category_code', how= 'left', suffixes=('_df1', '_df2'))
                columns_to_check = df.columns
                for col in columns_to_check:
                    if f"{col}_df1" in dfs.columns:
                        dfs[f"{col}"] = dfs[f"{col}_df1"].fillna(dfs[f"{col}_df2"])
                        dfs.drop(columns=[f"{col}_df1",f"{col}_df2"], inplace=True)    
            del df
                
            duration -= 1
            if start_month == 0:
                start_year -= 4
            else:
                start_year -= 1
        dfs = transpose_data(dfs, stock_code.upper())
        dfs.fillna(0, inplace=True)
        dfs['year'] = dfs['year'].astype(int)
        dfs['quarter'] = dfs['quarter'].astype(int)
        return fix_number_issue(dfs)  
    else:
        period = 1
        if start_month != 0:
            period = 2
            
        view = 3
        
        df = craw_24h_financial_statement(stock_code, period, view, duration)            
        df = map_sec_category_code(df,sheet)
        df['data'] = df['data'].astype(float)
        df['data'] = df['data']*1000
        df.drop_duplicates(subset=['category_code', 'year', 'quarter'], inplace=True)
        df.fillna(0, inplace=True)
        df['data'] = df['data'].astype(int)
        df['year'] = df['year'].astype(int)
        df['quarter'] = df['quarter'].astype(int)
        
        return df



class CafeFCrawlerFS(AbstractCrawlerFS):
    def __init__(self, args):
        super().__init__(args)
        self.sheet_types = ['bsheet', 'incsta', 'cashflow']
        self.sheet_types = ['income_statement' , 'balance_sheet', 'cash_flow_statement']
        self.start_year = 2024
        self.start_months = [0,2]
        self.durations = [2,6]
        
    def craw_bank_financial_report(self, **kwargs):
        
        # Assume fix duration for now
        is_bank = True
        dfs = []
        
        count = 1
        for sheet, sheet_type in zip(self.sheets, self.sheet_types):
            for start_month, duration in zip(self.start_months, self.durations):
                for stock_code in self.args.bank_stock_codes:
                    if count % 10 == 0:
                        time.sleep(5)
                        
                    count += 1
                    start_year_ = self.start_year
                    if start_month == 0:
                        start_year_ -=1
                    df = craw_cafef(stock_code, start_year_, start_month, duration, sheet, is_bank)
                    df['report_type'] = sheet_type
                    dfs.append(df)
                    
        dfs = pd.concat(dfs).reset_index(drop=True)
        logging.info(f"Bank financial report crawled with {dfs.shape[0]} records")
        test_df = dfs[(dfs['year'] == 2023) & (dfs['quarter'] == 0)]
        if test_df['stock_code'].nunique() != len(self.args.bank_stock_codes):
            logging.warning("Some stock codes are missing")
        return dfs
    
    def craw_non_bank_financial_report(self, **kwargs):
        is_bank = False
        dfs = []
        
        count = 1
        for sheet, sheet_type in zip(self.sheets, self.sheet_types):
            for start_month, duration in zip(self.start_months, self.durations):
                for stock_code in self.args.non_bank_stock_codes:
                    if count % 10 == 0:
                        time.sleep(5)
                        
                    count += 1
                    start_year_ = self.start_year
                    if start_month == 0:
                        start_year_ -=1
                    df = craw_cafef(stock_code, start_year_, start_month, duration, sheet, is_bank)
                    df['report_type'] = sheet_type
                    dfs.append(df)
                    
        dfs = pd.concat(dfs).reset_index(drop=True)
        logging.info(f"Non bank financial report crawled with {dfs.shape[0]} records")
        test_df = dfs[(dfs['year'] == 2023) & (dfs['quarter'] == 0)]
        if test_df['stock_code'].nunique() != len(self.args.non_bank_stock_codes):
            logging.warning("Some stock codes are missing")
        return dfs
    
    def craw_securities_financial_report(self, **kwargs):
        dfs = []
        
        count = 1
        for sheet, sheet_type in zip(self.sheets, self.sheet_types):
            for start_month, duration in zip(self.start_months, self.durations):
                for stock_code in self.args.securities_stock_codes:
                    if count % 5 == 0:
                        time.sleep(5)
                    count+=1
                    
                    df = crawl_mix_securities(stock_code, self.start_year, start_month, duration, sheet)
                    df['report_type'] = sheet_type
                    dfs.append(df)
        
        dfs = pd.concat(dfs).reset_index(drop=True)
        logging.info(f"Securities financial report crawled with {dfs.shape[0]} records")
        test_df = dfs[(dfs['year'] == 2023) & (dfs['quarter'] == 0)]
        if test_df['stock_code'].nunique() != len(self.args.securities_stock_codes):
            logging.warning("Some stock codes are missing")
        return dfs    
        