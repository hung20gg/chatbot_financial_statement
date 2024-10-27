import sys 
import os
current_path = os.path.dirname(__file__)
sys.path.append(current_path)

from collect_company_info import CrawCompanyInfo
from collect_financial_report import CrawFinancialReport
import ratio_index

class DataArgs:
    non_bank_stock_code = ["MWG", "NHA", "VNM", "HPG", "VHM", "PNJ", "YEG", "FPT","MSN", "GAS", "VRE", "VJC", "VIC", "PLX", "SAB", "POW", "GVR", "BCM", "VPI", "DVM", "KDH", "HDC", "TCH", "CEO", "HUT", "NVL", "DBC", "SAF", "DHT", "VTP", "PVT", "FRT", "DGC", "DCM", "NKG", "CMG", "VGI", "PVC", "CAP", "DTD", "HLD", "L14", "L18", "LAS", "LHC", "NTP", "PLC", "PSD", "PVG", "PVS", "SLS", "TIG", "TMB", "TNG", "TVD", "VC3", "VCS", "DXG"]
    bank_stock_code = ["BID", "EIB", "OCB", "CTG", "VCB", "ACB", "MBB", "HDB", "TPB", "VPB",  "STB", "TCB",  "SHB", "VIB", "CTG",  "ABB", "LPB", "NVB"]
    securities_stock_code = ["MBS", "VND", "SSI", "VIX", "ORS"]
    
class SimpleETL:
    def __init__(self, args):
        self.args = args