import sys 
import os

class AbstractCrawlerFS:
    def __init__(self, args):
        self.args = args
        
    def craw_bank_financial_report(self, **kwargs):
        raise NotImplementedError
    
    def craw_non_bank_financial_report(self, **kwargs):
        raise NotImplementedError
    
    def craw_securities_financial_report(self, **kwargs):
        raise NotImplementedError
    
    
class AbstractCrawlerCompany:
    def __init__(self, args):
        self.args = args
        
    def craw_company_info(self, stock_code, **kwargs):
        raise NotImplementedError
    
    def craw_shareholder_and_subsidiaries(self, stock_code, **kwargs):
        raise NotImplementedError