# Dictionary to hold functions and corresponding category codes
FINANCIAL_STRUCTURE_RATIO_FUNCTIONS = {
    'EBIT': ['IS_050', 'IS_023'],  # EBIT
    'equity_ratio': ['BS_400', 'BS_270'],  # equity, total_assets
    'long_term_asset_self_financing_ratio': [['BS_400', 'BS_330'], 'BS_200'],  # permanent_capital (equity + long_term_liabilities), long_term_assets
    'fixed_asset_self_financing_ratio': [['BS_400', 'BS_330'], ['BS_220','BS_240']],  # permanent_capital (equity + long_term_liabilities), fixed_assets
    'general_solvency_ratio': ['BS_270', 'BS_300'],  #  total_assets, total_liabilities
    'return_on_investment': ['IS_060', 'BS_270'],  # net_income, total_investment (using total assets as example)
    'ROIC': ['IS_061', ['BS_320', 'BS_338', 'BS_400']],  # NOPAT, invested_capital
    # 'return_on_long_term_capital': ['IS_050', 'BS_330'],  # EBIT, long_term_liabilities
    'basic_earning_power': [['IS_050', 'IS_023'], 'BS_270'],  # EBIT, total_assets
    'debt_to_assets_ratio': ['BS_300', 'BS_270'],  # total_liabilities, total_assets
    'debt_to_equity_ratio': ['BS_300', 'BS_400'],  # total_liabilities, equity
    'short_term_debt_to_assets_ratio': ['BS_310', 'BS_270'],  # short_term_liabilities, total_assets
    'interest_coverage_ratio': [['IS_050', 'IS_023'], 'IS_023'],  # EBIT, interest_expense
    'long_term_debt_to_equity_ratio': ['BS_330', 'BS_400'],  # long_term_liabilities, equity
    'short_term_debt_to_equity_ratio': ['BS_310', 'BS_400']  # short_term_liabilities, equity
}

LIQUIDITY_RATIO_FUNCTIONS = {
    'receivables_to_payables_ratio': [['BS_130', 'BS_210'], 'BS_300'],  # accounts_receivable, total_liabilities
    'receivables_to_total_assets_ratio': [['BS_130', 'BS_210'], 'BS_270'],  # accounts_receivable, total_assets
    'debt_to_total_capital_ratio': ['BS_300', 'BS_440'],  # total_liabilities, total_capital
    'receivables_to_sales_ratio': [['BS_131', 'BS_211'], 'IS_010'],  # accounts_receivables, total_sales
    'allowance_for_doubtful_accounts_ratio': [['BS_137', 'BS_219'], ['BS_131', 'BS_211']],  # allowance_for_doubtful_accounts, accounts_receivables
    'asset_to_debt_ratio': ['BS_270', 'BS_300'],  # total_assets, total_liabilities
    'current_ratio': [['BS_100', 'BS_151'], 'BS_310'],  # current_assets_for_liquidity, current_liabilities
    'quick_ratio': [['BS_100', 'BS_151'], 'BS_140', 'BS_310'],  # current_assets - inventory, current_liabilities
    'cash_ratio': ['BS_110', 'BS_310'],  # cash_and_cash_equivalents, current_liabilities
    'long_term_debt_coverage_ratio': ['BS_200', 'BS_330'],  # non_current_assets, non_current_liabilities
    'debt_to_equity_ratio': ['BS_300', 'BS_400'],  # total_liabilities, total_equity
    'long_term_debt_to_equity_capital_ratio': ['BS_330', 'BS_400'],  # long_term_liabilities, equity
    'time_interest_earned': [['IS_050', 'IS_023'], 'IS_023'],  # EBIT, interest_expense
    'debt_to_tangible_net_worth_ratio': ['BS_300', 'BS_400', 'BS_227'],  # total_liabilities, equity, intangible_assets
}

FINANCIAL_RATIO_FUNCTIONS = {
    'financial_leverage': ['BS_300', 'BS_440'],  # total_liabilities (BS_300), total_lia_and_equity (BS_440)
    'allowance_for_doubtful_accounts_to_total_assets_ratio': [['BS_137', 'BS_219'], 'BS_270'],  # allowance_for_doubtful_accounts (BS_137+BS_219), total_assets (BS_270)
    'permanent_financing_ratio': [['BS_400', 'BS_330'], 'BS_440'],  # permanent_capital (BS_400 + BS_330), total_lia_and_equity (BS_440)
}

INCOME_RATIO_FUNCTIONS = {
    'financial_income_to_net_revenue_ratio': ['IS_021', 'IS_010']  # financial_income (IS_021), net_revenue (IS_010)
}

PROFITABILITY_RATIO_FUNCTIONS = {
    'return_on_assets': ['IS_060', 'BS_270'],  # net_income (IS_060), total_assets (BS_270)
    'return_on_fixed_assets': ['IS_060', 'BS_220'],  # net_income (IS_060), average_fixed_assets (BS_220)
    'return_on_long_term_operating_assets':['IS_060', ['BS_240','BS_210','BS_220','BS_230','BS_260']],  # net_income (IS_060), average_long_term_operating_assets (BS_240)
    'Return_on_equity': ['IS_060', 'BS_400'],  # net_income (IS_060), equity (BS_400)
    # 'return_on_common_equity': ['IS_060', 'CF_036', 'BS_400'],  # net_income (IS_060), preferred_dividends (CF_036), average_common_equity (BS_400)
    'profitability_of_cost_of_goods_sold': ['IS_030', 'IS_011'],  # net_income_from_operating (IS_030), COGS (IS_011)
    'price_spread_ratio': ['IS_020', 'IS_011'],  # gross_profit (IS_020), COGS (IS_011)
    'profitability_of_operating_expenses': ['IS_030', ['IS_025', 'IS_026', 'IS_011']],  # net_income_from_operating (IS_030), total_operating_expenses (IS_025 + IS_026 + IS_011)
    'Return_on_sales': ['IS_060', 'IS_010'],  # net_income (IS_060), net_sales (IS_010)
    'operating_profit_margin': ['IS_030', 'IS_010'],  # NOPAT, net_sales (IS_010)
    'net_profit_margin': ['IS_060', 'IS_010'],  # net_income (IS_060), net_sales (IS_010)
    'gross_profit_margin': ['IS_020', 'IS_010'],  # gross_profit (IS_020), net_sales (IS_010)
    'return_on_assets_ttm': ['IS_060_TTM', 'BS_270'],  # net_income (IS_060), total_assets (BS_270)
    'return_on_equity_ttm': ['IS_060_TTM', 'BS_400'],  # net_income (IS_060), equity (BS_400)
}


CASHFLOW_RATIO_FUNCTIONS = {
    'EBITDA': [['IS_050', 'IS_023'], 'CF_002'],  # EBIT (IS_050), depreciation_and_amortization (CF_002)
    'free_cash_flow': ['CF_020', ['CF_021', 'CF_023'], 'CF_036'],  # operating_net_cash_flow (CF_020), capital_expenditures (CF_021 + CF_023), dividends_paid (CF_036)
    'free_cash_flow_to_operating_cash_flow_ratio': ['free_cash_flow', 'CF_020'],  # free_cash_flow, operating_net_cash_flow (CF_020)
    'cash_debt_coverage_ratio': ['CF_020', 'BS_300'],  # operating_net_cash_flow (CF_020), avg_total_liabilities (BS_300)
    'cash_interest_coverage': ['CF_020', 'IS_023'],  # operating_net_cash_flow (CF_020), interest_expense (IS_023)
    'cash_return_on_assets': ['CF_020', 'BS_270'],  # operating_net_cash_flow (CF_020), avg_total_assets (BS_270)
    'cash_return_on_fixed_assets': ['CF_020', 'BS_220'],  # operating_net_cash_flow (CF_020), avg_fixed_assets (BS_220)
    'CFO_to_total_equity': ['CF_020', 'BS_400'],  # operating_net_cash_flow (CF_020), avg_total_equity (BS_400)
    'cash_flow_from_sales_to_sales': ['CF_020', 'IS_010'],  # operating_net_cash_flow (CF_020), net_sales (IS_010)
    'cash_flow_margin': ['CF_020', ['IS_010', 'IS_021']],  # operating_net_cash_flow (CF_020), total_revenue (IS_010 + IS_021)
    'earning_quality_ratio': ['CF_020', 'IS_060'],  # operating_net_cash_flow (CF_020), net_income (IS_060)
}

CORP_AVG_RATIO_FUNCTIONS = {
    'return_on_average_assets': ['IS_060', 'BS_270'],  # net_income, avg_total_assets
    'return_on_average_equity': ['IS_060', 'BS_400'],  # net_income, avg_total_equity
    'return_on_average_sales':  ['IS_060', 'IS_010'],  # net_income, avg_sales
    'Total_Asset_Turnover': ['IS_010', 'BS_270'], # net_sales (IS_010), avg_total_assets (BS_270)
}

CORP_AVG_TTM_RATIO_FUNCTIONS = {
    'return_on_average_assets_ttm': ['IS_060_TTM', 'BS_270'],  # net_income, avg_total_assets
    'return_on_average_equity_ttm': ['IS_060_TTM', 'BS_400'],  # net_income, avg_total_equity
}

# YoY_RATIO_FUNCTIONS = {
#     'Net_Revenue_Growth_YoY': 'IS_010',
#     'Gross_Profit_Growth_YoY': 'IS_020',
#     'EBITDA_Growth_YoY': 'EBITDA',
#     'EBIT_Growth_YoY': 'EBIT',
#     'Pre_Tax_Profit_Growth_YoY': 'IS_050',
#     'Accounts_Receivable_Growth_YoY': ['BS_131', 'BS_211'],  # Summing up these codes
#     'Inventory_Growth_YoY': 'BS_140',
#     'Short_Term_Debt_Growth_YoY': 'BS_320',
#     'Long_Term_Debt_Growth_YoY': 'BS_338',
#     'SG&A_Expense_Growth_YoY': ['IS_025', 'IS_026'],  # Summing up these codes
#     'Total_Asset_Growth_YoY': 'BS_270',
#     'Equity_Growth_YoY': 'BS_400',
#     'CFO_Growth_YoY': 'CF_020'
# }

# BANK #

BANK_FINANCIAL_STRUCTURE_RATIO_FUNCTIONS = {
    'EBIT': ['IS_017', 'IS_002'],  # EBIT, interest_expense
    'equity_ratio': ['BS_500', 'BS_300'],  # equity, total_assets
    'long_term_asset_self_financing_ratio': ['BS_500', ['BS_220','BS_210','BS_240']],  # permanent_capital (equity ), long_term_assets ( fixed assets, Long-term capital investments, property investments)
    'fixed_asset_self_financing_ratio': ['BS_500', 'BS_220'],  # permanent_capital (equity ), fixed_assets
    'general_solvency_ratio': ['BS_300', 'BS_400'],  #  total_assets, total_liabilities
    'return_on_investment': ['IS_021', 'BS_300'],  # net_income, total_investment (using total assets as example)
    'basic_earning_power': [['IS_017', 'IS_002'], 'BS_300'],  # EBIT, total_assets
    'debt_to_assets_ratio': ['BS_400', 'BS_300'],  # total_liabilities, total_assets
    'debt_to_equity_ratio': ['BS_400', 'BS_500'],  # total_liabilities, equity
    'interest_coverage_ratio': [['IS_017', 'IS_002'], 'IS_002'],  # EBIT, interest_expense
    'long_term_debt_to_equity_ratio': [['BS_350','BS_360','BS_372'], 'BS_500'],  # long_term_liabilities(BS_350 + BS_360 + BS_372), equity
}


BANK_LIQUIDITY_RATIO_FUNCTIONS = {
    'receivables_to_payables_ratio': [ 'BS_251', 'BS_400'],  # accounts_receivable, total_liabilities
    'receivables_to_total_assets_ratio': [ 'BS_251', 'BS_300'],  # accounts_receivable, total_assets
    'debt_to_total_capital_ratio': ['BS_400', ['BS_400','BS_500']],  # total_liabilities, total_capital
    'receivables_to_sales_ratio': ['BS_251', ['IS_010','IS_001','IS_004']],  # accounts_receivables, total_sales
    'allowance_for_loan_customers_ratio': [ 'BS_169', 'BS_161'],  # allowance_for_loan_customers, loan_to_customers
    'asset_to_debt_ratio': ['BS_300', 'BS_400'],  # total_assets, total_liabilities
    'current_ratio': [[ 'BS_110', 'BS_120', 'BS_131', 'BS_132'],['BS_370','BS_371','BS_373']],  # current_assets_for_liquidity(current_assets = BS_110 + BS_120 + BS_131 + BS_132), current_liabilities(current_liabilities = BS_370 + BS_371 + BS_373)
    # 'quick_ratio': [['BS_100', 'BS_151'], 'BS_140', 'BS_310'],  # current_assets - inventory, current_liabilities
    'cash_ratio': [['BS_110','BS_120'], ['BS_370','BS_371','BS_373']],  # cash_and_cash_equivalents, current_liabilities
    'long_term_debt_coverage_ratio': [['BS_210','BS_220','BS_240'], ['BS_350','BS_360']],  # non_current_assets, non_current_liabilities
    'debt_to_equity_ratio': ['BS_400', 'BS_500'],  # total_liabilities, total_equity
    'long_term_debt_to_equity_capital_ratio': [['BS_350','BS_360'], 'BS_500'],  # long_term_liabilities, equity
    'time_interest_earned': [['IS_017', 'IS_002'], 'IS_002'],  # EBIT, interest_expense
    'debt_to_tangible_net_worth_ratio': ['BS_400', 'BS_500', 'BS_227'],  # total_liabilities, equity, intangible_assets
    'cost_to_income_ratio': ['IS_014', ['IS_006', 'IS_007', 'IS_008','IS_009', 'IS_012', 'IS_013']],  # total_operating_expenses, net_sales
}   


BANK_FINANCIAL_RATIO_FUNCTIONS = {
    'financial_leverage': ['BS_400', 'BS_800'],  # total_liabilities (BS_300), total_lia_and_equity 
    'allowance_for_doubtful_accounts_to_total_assets_ratio': ['BS_169', 'BS_300'],  # allowance_for_doubtful_accounts , total_assets 
}

BANK_INCOME_RATIO_FUNCTIONS = {
    'financial_income_to_net_revenue_ratio': [['IS_004','IS_007','IS_008','IS_009','IS_010'], ['IS_003','IS_004','IS_007','IS_008','IS_009','IS_010']]  # financial_income (IS_004 + IS_007 + IS_008 + IS_009 + IS_010), net_revenue (IS_010)
}

BANK_PROFITABILITY_RATIO_FUNCTIONS = {
    'return_on_assets': ['IS_021', 'BS_300'],  # net_income , total_assets 
    'return_on_fixed_assets': ['IS_021', 'BS_220'],  # net_income , average_fixed_assets 
    'return_on_long_term_operating_assets': ['IS_021', ['BS_210','BS_220','BS_240']],  # net_income , average_long_term_operating_assets 
    'Return_on_equity': ['IS_021', 'BS_500'],  # net_income , equity 
    # 'return_on_common_equity': ['IS_060', 'CF_036', 'BS_400'],  # net_income (IS_060), preferred_dividends (CF_036), average_common_equity (BS_400)
    'profitability_of_cost_of_goods_sold': ['IS_015', 'IS_005'],  # net_income_from_operating (IS_030), COGS (IS_011)
    # 'price_spread_ratio': ['IS_020', 'IS_011'],  # gross_profit (IS_020), COGS (IS_011)
    'profitability_of_operating_expenses': ['IS_017', 'IS_014'],  # net_income_from_operating , total_operating_expenses 
    'Return_on_sales': ['IS_021', ['IS_010','IS_003','IS_004']],  # net_income , net_sales 
    'net_profit_margin': ['IS_021', 'IS_001'],  # net_income , net_sales
    'operating_profit_margin': ['IS_001','IS_014', ['IS_010','IS_003','IS_004']],  # NOPAT, net_sales 
    'gross_profit_margin': ['IS_003', 'IS_001'],  # gross_profit , net_sales 
    'return_on_assets_ttm': ['IS_021_TTM', 'BS_300'],  # net_income (IS_060), total_assets (BS_270)
    'return_on_equity_ttm': ['IS_021_TTM', 'BS_500'],  # net_income (IS_060), equity (BS_400)
}

BANK_CASHFLOW_RATIO_FUNCTIONS = {
    'EBITDA': [['IS_017', 'IS_002'],[ 'BS_223','BS_226','BS_229','BS_242']],  # EBIT , depreciation_and_amortization 
    'free_cash_flow': ['CF_024', 'CF_025', 'CF_038'],  # operating_net_cash_flow , capital_expenditures , dividends_paid 
    'free_cash_flow_to_operating_cash_flow_ratio': ['free_cash_flow', 'CF_024'],  # free_cash_flow, operating_net_cash_flow 
    'cash_debt_coverage_ratio': ['CF_024', 'BS_400'],  # operating_net_cash_flow , avg_total_liabilities 
    'cash_interest_coverage': ['CF_024', 'IS_002'],  # operating_net_cash_flow , interest_expense
    'cash_return_on_assets': ['CF_024', 'BS_300'],  # operating_net_cash_flow , avg_total_assets 
    'cash_return_on_fixed_assets': ['CF_024', 'BS_220'],  # operating_net_cash_flow , avg_fixed_assets 
    'CFO_to_total_equity': ['CF_024', 'BS_500'],  # operating_net_cash_flow , avg_total_equity 
    'cash_flow_from_sales_to_sales': ['CF_024', ['IS_010','IS_003','IS_004']],  # operating_net_cash_flow , net_sales 
    'cash_flow_margin': ['CF_024', ['IS_010','IS_003','IS_004','IS_007','IS_008','IS_009']],  # operating_net_cash_flow , total_revenue 
    'earning_quality_ratio': ['CF_024', 'IS_021'],  # operating_net_cash_flow , net_income 
    'net_interest_margin': ['IS_003', ['BS_161','BS_130','BS_170']],  # net_interest_income , avg_earning_assets
}


BANK_FIIN_RATIO_FUNCTIONS = {
    'current_account_saving_account_ratio':  ['BS_330','Bank_TM_121','Bank_TM_124'],  
    'bad_debt_ratio': ['BS_161', ['Bank_TM_68', 'Bank_TM_69', 'Bank_TM_70' ]],
    'non_performing_loan_coverage_ratio': ['BS_169', 'Bank_TM_65'],
}

BANK_AVG_RATIO_FUNCTIONS = {
    'return_on_average_assets': ['IS_021', 'BS_300'],  # net_income, avg_total_assets
    'return_on_average_equity': ['IS_021', 'BS_500'],  # net_income, avg_total_equity
    'return_on_average_sales':   ['IS_021', ['IS_010','IS_003','IS_004']],  # net_income, avg_sales
    'Basic_Earning_Power_Ratio': [['IS_017', 'IS_002'], 'BS_300'],  # EBIT , total_assets 
}
BANK_AVG_TTM_RATIO_FUNCTIONS = {
    'return_on_average_assets_ttm': ['IS_021_TTM', 'BS_300'],
    'return_on_average_equity_ttm': ['IS_021_TTM', 'BS_500'],
}

# BANK_YoY_RATIO_FUNCTIONS = {
#     'Customer_Deposits_Growth_YoY' : 'BS_330',
#     'Operating_Expense_Growth_YoY' : 'IS_014',
#     'Income_Before_Provision_Growth_YoY' : 'IS_015',
#     'Interest_Income_Growth_YoY' : 'IS_003',
#     'Non_Interest_Income_Growth_YoY' : ['IS_006','IS_007','IS_008','IS_009','IS_012','IS_013'],
#     'Total_Operating_Income_Growth' : 'IS_030',
#     'Customer_Loans_Growth_YoY' : 'BS_160',
# }

# SECURITIES #

SECURITIES_FINANCIAL_STRUCTURE_RATIO_FUNCTIONS = {
    'EBIT': ['IS_090', 'IS_052'],  # EBIT, interest_expense
    'equity_ratio': ['BS_400', 'BS_270'],  # equity, total_assets
    'long_term_asset_self_financing_ratio': [['BS_410','BS_340'], 'BS_200'],  # permanent_capital (equity+long-term liabilities ), long_term_assets (non-current assets)
    'fixed_asset_self_financing_ratio': [['BS_410','BS_340'], 'BS_220'],  # permanent_capital (equity ), fixed_assets
    'general_solvency_ratio': ['BS_270', 'BS_300'],  #  total_assets, total_liabilities
    'return_on_investment': ['IS_200', 'BS_212'],  # net_income, total_investment (using total assets as example)
    'ROIC': ['IS_100', ['BS_400','BS_312','BS_346']],  # EBIT,Tax_expense, invested_capital
    # 'return_on_long_term_capital': [['IS_090','IS_052'], ['BS_410','BS_340']],  # EBIT, avg_long_term_capital
    'basic_earning_power': [['IS_090','IS_052'], 'BS_270'],  # EBIT, avg_total_assets
    'debt_to_assets_ratio': ['BS_300', 'BS_270'],  # total_liabilities, total_assets
    'debt_to_equity_ratio': ['BS_300', 'BS_410'],  # total_liabilities, equity
    'short_term_debt_to_assets_ratio': ['BS_310', 'BS_270'],  # short_term_liabilities, total_assets
    'interest_coverage_ratio': [['IS_090','IS_052'], 'IS_052'],  # EBIT, interest_expense
    'long_term_debt_to_equity_ratio': ['BS_340', 'BS_410'],  # long_term_liabilities(BS_350 + BS_360 + BS_372), equity
    'short_term_debt_to_equity_ratio': ['BS_310', 'BS_410']  # short_term_liabilities, equity
}


SECURITIES_LIQUIDITY_RATIO_FUNCTIONS = {
    'receivables_to_payables_ratio': [ 'BS_117', 'BS_300'],  # accounts_receivable, total_liabilities
    'receivables_to_total_assets_ratio': [ 'BS_117', 'BS_270'],  # accounts_receivable, total_assets
    'debt_to_total_capital_ratio': ['BS_300', ['BS_411']],  # total_liabilities, total_capital
    'receivables_to_sales_ratio': ['BS_117', ['IS_010','IS_020','IS_050','IS_006','IS_007','IS_008','IS_009']],  # accounts_receivables, total_sales
    'asset_to_debt_ratio': ['BS_270', 'BS_300'],  # total_assets, total_liabilities
    'current_ratio': ['BS_100','BS_310'],  # current_assets_for_liquidity(), current_liabilities
    # 'quick_ratio': [['BS_100', 'BS_151'], 'BS_140', 'BS_310'],  # current_assets - inventory, current_liabilities
    'cash_ratio': ['BS_111', 'BS_310'],  # cash_and_cash_equivalents, current_liabilities
    'long_term_debt_coverage_ratio': ['BS_200', 'BS_340'],  # non_current_assets, non_current_liabilities
    'debt_to_equity_ratio': ['BS_300', 'BS_400'],  # total_liabilities, total_equity
    'long_term_debt_to_equity_capital_ratio': ['BS_340', 'BS_400'],  # long_term_liabilities, equity
    'time_interest_earned': ['IS_090', 'IS_052'],  # EBIT, interest_expense
    'debt_to_tangible_net_worth_ratio': ['BS_300', 'BS_400', 'BS_227'],  # total_liabilities, equity, intangible_assets
}


SECURITIES_FINANCIAL_RATIO_FUNCTIONS = {
    'financial_leverage': ['BS_300', 'BS_440'],  # total_liabilities , total_lia_and_equity 
    'allowance_for_doubtful_accounts_to_total_assets_ratio': ['BS_129', 'BS_270'],  # allowance_for_doubtful_accounts , total_assets 
    # 'permanent_financing_ratio': [, 'BS_440'],  # permanent_capital (BS_400 + BS_330), total_lia_and_equity 
}


SECURITIES_INCOME_RATIO_FUNCTIONS = {
    'financial_income_to_net_revenue_ratio': ['IS_050','IS_020']
}


SECURITIES_PROFITABILITY_RATIO_FUNCTIONS = {
    'return_on_assets': ['IS_200', 'BS_270'],  # net_income , total_assets 
    'return_on_fixed_assets': ['IS_200', 'BS_220'],  # net_income , average_fixed_assets 
    'return_on_long_term_operating_assets': ['IS_200', ['BS_211','BS_220','BS_230','BS_240','BS_250']],  # net_income , average_long_term_operating_assets 
    'Return_on_equity': ['IS_200', 'BS_400'],  # net_income , equity 
    'profitability_of_operating_expenses': ['IS_200','IS_040', 'IS_040'],  # net_income_from_operating , total_operating_expenses 
    'Return_on_sales': ['IS_200', 'IS_020'],  # net_income , net_sales 
    'net_profit_margin': ['IS_200', 'IS_020'],  # net_income , net_sales
    'operating_profit_margin': ['IS_200','IS_040', 'IS_020'],  # NOPAT, net_sales 
    'gross_profit_margin': ['IS_040.1', 'IS_020'],  # gross_profit , net_sales 
    'Total_Asset_Turnover': ['IS_020', 'BS_270'],  # net_sales , avg_total_assets
    'return_on_assets_ttm': ['IS_200_TTM', 'BS_270'],  # net_income , total_assets
    'return_on_equity_ttm': ['IS_200_TTM', 'BS_400'],  # net_income , equity
}


SECURITIES_CASHFLOW_RATIO_FUNCTIONS = {
    'EBITDA': [['IS_090','IS_052'], 'CF_003'],  # EBIT (IS_050), depreciation_and_amortization (CF_002)
    'free_cash_flow': ['CF_060', 'CF_061', None],  # operating_net_cash_flow , capital_expenditures , dividends_paid 
    'free_cash_flow_to_operating_cash_flow_ratio': ['free_cash_flow', 'CF_060'],  # free_cash_flow, operating_net_cash_flow 
    'cash_debt_coverage_ratio': ['CF_060', 'BS_300'],  # operating_net_cash_flow , avg_total_liabilities 
    'cash_interest_coverage': ['CF_060', 'IS_052'],  # operating_net_cash_flow , interest_expense
    'cash_return_on_assets': ['CF_060', 'BS_270'],  # operating_net_cash_flow , avg_total_assets 
    'cash_return_on_fixed_assets': ['CF_060', 'BS_220'],  # operating_net_cash_flow , avg_fixed_assets 
    'CFO_to_total_equity': ['CF_060', 'BS_400'],  # operating_net_cash_flow , avg_total_equity 
    'cash_flow_from_sales_to_sales': ['CF_060', 'IS_020'],  # operating_net_cash_flow , net_sales 
    'cash_flow_margin': ['CF_060', 'IS_020'],  # operating_net_cash_flow , total_revenue 
    'earning_quality_ratio': ['CF_060', 'IS_200'],  # operating_net_cash_flow , net_income 
    'net_interest_margin': ['IS_001', 'BS_110'],  # net_interest_income , avg_earning_assets
}

SECURITIES_AVG_RATIO_FUNCTIONS = {
    'return_on_average_assets': ['IS_200', 'BS_270'],  # net_income, avg_total_assets
    'return_on_average_equity': ['IS_200', 'BS_400'],  # net_income, avg_total_equity
    'return_on_average_sales':   ['IS_200', 'IS_020'],  # net_income, avg_sales
    'Basic_Earning_Power_Ratio': [['IS_090','IS_052'], 'BS_270'],  # EBIT , total_assets 
}

SECURITIES_AVG_TTM_RATIO_FUNCTIONS = {
    'return_on_average_assets_ttm': ['IS_200_TTM', 'BS_270'],
    'return_on_average_equity_ttm': ['IS_200_TTM', 'BS_400'],
}


PE_RATIO_FUNCTIONS = {
    'earning_per_share': ['IS_070'],
    'price_earning_ratio': ['Price', 'IS_070'],
    'book_value_per_share': ['IS_070','IS_060', 'BS_400'],
    'price_to_book_ratio': ['Price', 'IS_070','IS_060', 'BS_400'],
}

BANK_PE_RATIO_FUNCTIONS = {
    'earning_per_share': ['IS_023'],
    'price_earning_ratio': ['Price', 'IS_023'],
    'book_value_per_share': ['IS_023','IS_021', 'BS_500'],
    'price_to_book_ratio': ['Price', 'IS_023','IS_021', 'BS_500'],
}

SECURITIES_PE_RATIO_FUNCTIONS = {
    'earning_per_share': ['IS_501'],
    'price_earning_ratio': ['Price', 'IS_501'],
    'book_value_per_share': ['IS_501','IS_200', 'BS_400'],
    'price_to_book_ratio': ['Price', 'IS_501','IS_200', 'BS_400'],
}


DATE_RELATED_FUNCTIONS = {
    'days_sales_outstanding' : ['BS_130', 'IS_010'],
    'days_payable_outstanding' : ['BS_311', 'BS_331', 'IS_011'],
    'days_inventory_outstanding' : ['BS_140', 'IS_011'],
    'cash_conversion_cycle' : ['BS_130', 'BS_311', 'BS_331', 'BS_140', 'IS_011', 'IS_010'],
}

BANK_DATE_RELATED_FUNCTIONS = {
}

SECURITIES_DATE_RELATED_FUNCTIONS = {
    'days_sales_outstanding' : ['BS_117', 'IS_020'],
    'days_payable_outstanding' : ['BS_320', 'BS_347', 'IS_040'],
}

YoY_RATIO_FUNCTIONS = {
    'securities':{
        'Net_Revenue_Growth_YoY': 'IS_020',
        'Gross_Profit_Growth_YoY': 'IS_040.1',
        'Net_Income_Growth_YoY': 'IS_200',
        'EBITDA_Growth_YoY': 'EBITDA',
        'EBIT_Growth_YoY': 'EBIT',
        'Pre_Tax_Profit_Growth_YoY': 'IS_090',
        'Accounts_Receivable_Growth_YoY': 'BS_117',
        'Short_Term_Debt_Growth_YoY': 'BS_310',
        'Long_Term_Debt_Growth_YoY': 'BS_330',
        'SG&A_Expense_Growth_YoY': ['IS_061', 'IS_062'],  # Summing up these codes
        'Total_Asset_Growth_YoY': 'BS_270',
        'Equity_Growth_YoY': 'BS_400',
        'Liability_Growth_YoY': 'BS_300',
        'CFO_Growth_YoY': 'CF_060'
        },
    'non_bank':{

        'Net_Revenue_Growth_YoY': 'IS_010',
        'Gross_Profit_Growth_YoY': 'IS_020',
        'Net_Income_Growth_YoY': 'IS_060',
        'EBITDA_Growth_YoY': 'EBITDA',
        'EBIT_Growth_YoY': 'EBIT',
        'Pre_Tax_Profit_Growth_YoY': 'IS_050',
        'Accounts_Receivable_Growth_YoY': ['BS_131', 'BS_211'],  # Summing up these codes
        'Inventory_Growth_YoY': 'BS_140',
        'Short_Term_Debt_Growth_YoY': 'BS_310',
        'Long_Term_Debt_Growth_YoY': 'BS_340',
        'SG&A_Expense_Growth_YoY': ['IS_025', 'IS_026'],  # Summing up these codes
        'Total_Asset_Growth_YoY': 'BS_270',
        'Equity_Growth_YoY': 'BS_400',
        'Liability_Growth_YoY': 'BS_300',
        'CFO_Growth_YoY': 'CF_020'
    },
    'bank':{
        'Net_Income_Growth_YoY' : 'IS_021',
        'Customer_Deposits_Growth_YoY' : 'BS_330',
        'Operating_Expense_Growth_YoY' : 'IS_014',
        'Income_Before_Provision_Growth_YoY' : 'IS_015',
        'Interest_Income_Growth_YoY' : 'IS_003',
        'Non_Interest_Income_Growth_YoY' : ['IS_006','IS_007','IS_008','IS_009','IS_012','IS_013'],
        'Total_Operating_Income_Growth' : 'IS_030',
        'Customer_Loans_Growth_YoY' : 'BS_160',
        'Total_Asset_Growth_YoY' : 'BS_300',
        'Equity_Growth_YoY': 'BS_500',
        'outstanding_credit_balance_YoY' : 'BS_200',
        }

}
QoQ_RATIO_FUNCTIONS = {
    'securities': {
        'Net_Revenue_Growth_QoQ': 'IS_020',
        'Gross_Profit_Growth_QoQ': 'IS_040.1',
        'Net_Income_Growth_QoQ': 'IS_200',
        'EBITDA_Growth_QoQ': 'EBITDA',
        'EBIT_Growth_QoQ': 'EBIT',
        'Pre_Tax_Profit_Growth_QoQ': 'IS_090',
        'Accounts_Receivable_Growth_QoQ': 'BS_117',
        'Short_Term_Debt_Growth_QoQ': 'BS_310',
        'Long_Term_Debt_Growth_QoQ': 'BS_330',
        'SG&A_Expense_Growth_QoQ': ['IS_061', 'IS_062'],
        'Total_Asset_Growth_QoQ': 'BS_270',
        'Equity_Growth_QoQ': 'BS_400',
        'Liability_Growth_QoQ': 'BS_300',
        'CFO_Growth_QoQ': 'CF_060'
    },
    'non_bank': {
        'Net_Revenue_Growth_QoQ': 'IS_010',
        'Gross_Profit_Growth_QoQ': 'IS_020',
        'Net_Income_Growth_QoQ': 'IS_060',
        'EBITDA_Growth_QoQ': 'EBITDA',
        'EBIT_Growth_QoQ': 'EBIT',
        'Pre_Tax_Profit_Growth_QoQ': 'IS_050',
        'Accounts_Receivable_Growth_QoQ': ['BS_131', 'BS_211'],
        'Inventory_Growth_QoQ': 'BS_140',
        'Short_Term_Debt_Growth_QoQ': 'BS_310',
        'Long_Term_Debt_Growth_QoQ': 'BS_340',
        'SG&A_Expense_Growth_QoQ': ['IS_025', 'IS_026'],
        'Total_Asset_Growth_QoQ': 'BS_270',
        'Equity_Growth_QoQ': 'BS_400',
        'Liability_Growth_QoQ': 'BS_300',
        'CFO_Growth_QoQ': 'CF_020'
    },
    'bank': {
        'Net_Income_Growth_QoQ' : 'IS_021',
        'Customer_Deposits_Growth_QoQ' : 'BS_330',
        'Operating_Expense_Growth_QoQ' : 'IS_014',
        'Income_Before_Provision_Growth_QoQ' : 'IS_015',
        'Interest_Income_Growth_QoQ' : 'IS_003',
        'Non_Interest_Income_Growth_QoQ' : ['IS_006','IS_007','IS_008','IS_009','IS_012','IS_013'],
        'Total_Operating_Income_Growth_QoQ' : 'IS_030',
        'Customer_Loans_Growth_QoQ' : 'BS_160',
        'Total_Asset_Growth_QoQ' : 'BS_300',
        'Equity_Growth_QoQ': 'BS_500',
        'Pre_Tax_Profit_Growth_QoQ':'IS_017'

}

}

INDUSTRIES_TRANSLATION = {
    "Ngân hàng": "Banking",
    "Bất động sản": "Real Estate",
    "Xây dựng và Vật liệu": "Construction and Materials",
    "Thực phẩm và đồ uống": "Food and Beverages",
    "Hóa chất": "Chemicals",
    "Dịch vụ tài chính": "Financial Services",
    "Tài nguyên Cơ bản": "Basic Resources",
    "Bán lẻ": "Retail",
    "Dầu khí": "Oil and Gas",
    "Điện, nước & xăng dầu khí đốt": "Utilities (Electricity, Water & Gas)",
    "Hàng cá nhân & Gia dụng": "Personal and Household Goods",
    "Công nghệ Thông tin": "Information Technology",
    "Hàng & Dịch vụ Công nghiệp": "Industrial Goods and Services",
    "Truyền thông": "Media",
    "Du lịch và Giải trí": "Travel and Leisure",
    "Y tế": "Healthcare",
    "Ô tô và phụ tùng": "Automobiles and Parts",
    "Viễn thông": "Telecommunications"
}


IGNORE_TTM_CODES_IS = ['IS_022', 'IS_023', 'IS_024', 'IS_025', 'IS_081', 'IS_082', 'IS_046', 'IS_047', 'IS_048', 'IS_041', 'IS_024', 'IS_004', 'IS_005', 'IS_006', 'IS_007', 'IS_110', 'IS_111', 'IS_112', 'IS_120', 'IS_101', 'IS_102']
IGNORE_TTM_CODES_CF = ['CF_002', 'CF_009', 'CF_010', 'CF_011', 'CF_021', 'CF_080', 'CF_081', 'CF_057', 'CF_058', 'CF_120', 
 'CF_021', 'CF_022', 'CF_023', 'CF_024', 'CF_025', 'CF_026', 'CF_027', 'CF_028', 'CF_029', 'CF_030', 
 'CF_031', 'CF_032', 'CF_033', 'CF_200', 'CF_201', 'CF_202', 'CF_203', 'CF_204', 'CF_205', 'CF_206', 
 'CF_207', 'CF_210', 'CF_211', 'CF_212', 'CF_213', 'CF_034', 'CF_035', 'CF_036', 'CF_037', 'CF_038', 
 'CF_039', 'CF_040', 'CF_041', 'CF_042', 'CF_043']