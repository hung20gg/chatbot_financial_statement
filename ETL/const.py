# Dictionary to hold functions and corresponding category codes
FINANCIAL_STRUCTURE_RATIO_FUNCTIONS = {
    'equity_ratio': ['BS_400', 'BS_270'],  # equity, total_assets
    'long_term_asset_self_financing_ratio': [['BS_400', 'BS_330'], 'BS_200'],  # permanent_capital (equity + long_term_liabilities), long_term_assets
    'fixed_asset_self_financing_ratio': [['BS_400', 'BS_330'], ['BS_220','BS_240']],  # permanent_capital (equity + long_term_liabilities), fixed_assets
    'general_solvency_ratio': ['BS_270', 'BS_300'],  #  total_assets, total_liabilities
    'return_on_investment': ['IS_060', 'BS_270'],  # net_income, total_investment (using total assets as example)
    # 'ROIC': ['IS_060', 'BS_200'],  # NOPAT, invested_capital
    # 'return_on_long_term_capital': ['IS_050', 'BS_330'],  # EBIT, long_term_liabilities
    'basic_earning_power': [['IS_050','IS_023'], 'BS_270'],  # EBIT, total_assets
    'debt_to_assets_ratio': ['BS_300', 'BS_270'],  # total_liabilities, total_assets
    'debt_to_equity_ratio': ['BS_300', 'BS_400'],  # total_liabilities, equity
    'short_term_debt_to_assets_ratio': ['BS_310', 'BS_270'],  # short_term_liabilities, total_assets
    'interest_coverage_ratio': [['IS_050','IS_023'], 'IS_023'],  # EBIT, interest_expense
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
    'return_on_long_term_operating_assets': ['IS_060', 'BS_240'],  # net_income (IS_060), average_long_term_operating_assets (BS_240)
    'Basic_Earning_Power_Ratio': [['IS_050', 'IS_023'], 'BS_270'],  # EBIT (IS_050 + IS_023), total_assets (BS_270)
    'Return_on_equity': ['IS_060', 'BS_400'],  # net_income (IS_060), equity (BS_400)
    # 'return_on_common_equity': ['IS_060', 'CF_036', 'BS_400'],  # net_income (IS_060), preferred_dividends (CF_036), average_common_equity (BS_400)
    'profitability_of_cost_of_goods_sold': ['IS_030', 'IS_011'],  # net_income_from_operating (IS_030), COGS (IS_011)
    'price_spread_ratio': ['IS_020', 'IS_011'],  # gross_profit (IS_020), COGS (IS_011)
    'profitability_of_operating_expenses': ['IS_030', ['IS_025', 'IS_026', 'IS_011']],  # net_income_from_operating (IS_030), total_operating_expenses (IS_025 + IS_026 + IS_011)
    'Return_on_sales': ['IS_060', 'IS_010'],  # net_income (IS_060), net_sales (IS_010)
    'operating_profit_margin': ['IS_030', 'IS_010'],  # NOPAT, net_sales (IS_010)
    'gross_profit_margin': ['IS_020', 'IS_010'],  # gross_profit (IS_020), net_sales (IS_010)
}


CASHFLOW_RATIO_FUNCTIONS = {
    'EBITDA': ['IS_050', 'CF_002'],  # EBIT (IS_050), depreciation_and_amortization (CF_002)
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


# BANK #

BANK_FINANCIAL_STRUCTURE_RATIO_FUNCTIONS = {
    'equity_ratio': ['BS_500', 'BS_300'],  # equity, total_assets
    'long_term_asset_self_financing_ratio': ['BS_500', ['BS_220','BS_210','BS_240']],  # permanent_capital (equity ), long_term_assets ( fixed assets, Long-term capital investments, property investments)
    'fixed_asset_self_financing_ratio': ['BS_500', 'BS_220'],  # permanent_capital (equity ), fixed_assets
    'general_solvency_ratio': ['BS_300', 'BS_400'],  #  total_assets, total_liabilities
    'return_on_investment': ['IS_021', 'BS_300'],  # net_income, total_investment (using total assets as example)
    'basic_earning_power': ['IS_017', 'BS_300'],  # EBIT, total_assets
    'debt_to_assets_ratio': ['BS_400', 'BS_300'],  # total_liabilities, total_assets
    'debt_to_equity_ratio': ['BS_400', 'BS_500'],  # total_liabilities, equity
    'interest_coverage_ratio': ['IS_017', 'IS_002'],  # EBIT, interest_expense
    'long_term_debt_to_equity_ratio': [['BS_350','BS_360','BS_372'], 'BS_500'],
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
    'time_interest_earned': ['IS_017', 'IS_002'],  # EBIT, interest_expense
    'debt_to_tangible_net_worth_ratio': ['BS_400', 'BS_500', 'BS_227'],  # total_liabilities, equity, intangible_assets
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
    'Basic_Earning_Power_Ratio': ['IS_017', 'BS_300'],  # EBIT , total_assets 
    'Return_on_equity': ['IS_021', 'BS_500'],  # net_income , equity 
    # 'return_on_common_equity': ['IS_060', 'CF_036', 'BS_400'],  # net_income (IS_060), preferred_dividends (CF_036), average_common_equity (BS_400)
    # 'profitability_of_cost_of_goods_sold': ['IS_030', 'IS_011'],  # net_income_from_operating (IS_030), COGS (IS_011)
    # 'price_spread_ratio': ['IS_020', 'IS_011'],  # gross_profit (IS_020), COGS (IS_011)
    'profitability_of_operating_expenses': ['IS_017', 'IS_014'],  # net_income_from_operating , total_operating_expenses 
    'Return_on_sales': ['IS_021', ['IS_010','IS_003','IS_004']],  # net_income , net_sales 
    'operating_profit_margin': [['IS_010','IS_003','IS_004'],'IS_014', ['IS_010','IS_003','IS_004']],  # NOPAT, net_sales 
    # 'gross_profit_margin': ['IS_020', ['IS_010','IS_003','IS_004','IS_007','IS_008','IS_009']],  # gross_profit , net_sales 
}

BANK_CASHFLOW_RATIO_FUNCTIONS = {
    # 'EBITDA': ['IS_050', 'CF_002'],  # EBIT (IS_050), depreciation_and_amortization (CF_002)
    'free_cash_flow': ['CF_024', 'CF_025', 'CF_038'],  # operating_net_cash_flow , capital_expenditures , dividends_paid 
    'free_cash_flow_to_operating_cash_flow_ratio': ['free_cash_flow', 'CF_024'],  # free_cash_flow, operating_net_cash_flow 
    'cash_debt_coverage_ratio': ['CF_024', 'BS_400'],  # operating_net_cash_flow , avg_total_liabilities 
    'cash_interest_coverage': ['CF_024', 'IS_002'],  # operating_net_cash_flow , interest_expense
    'cash_return_on_assets': ['CF_024', 'BS_300'],  # operating_net_cash_flow , avg_total_assets 
    'cash_return_on_fixed_assets': ['CF_024', 'BS_220'],  # operating_net_cash_flow , avg_fixed_assets 
    'CFO_to_total_equity': ['CF_024', 'BS_500'],  # operating_net_cash_flow , avg_total_equity 
    'cash_flow_from_sales_to_sales': ['CF_024', ['IS_010','IS_003','IS_004']],  # operating_net_cash_flow , net_sales 
    'cash_flow_margin': ['CF_020', ['IS_010','IS_003','IS_004','IS_007','IS_008','IS_009']],  # operating_net_cash_flow , total_revenue 
    'earning_quality_ratio': ['CF_024', 'IS_021'],  # operating_net_cash_flow , net_income 
    'net_interest_margin': ['IS_003', ['BS_161','BS_130','BS_170']],  # net_interest_income , avg_earning_assets
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