import sys 
import os
current_path = os.path.dirname(__file__)
sys.path.append(current_path)


import const

import pandas as pd 
import numpy as np 


# Code modified from @pphanhh

#====================#
#   Utils Function   #
#====================#

def __get_financial_ratio(data_df, ratio_function):
    pivot_df = data_df.pivot_table(index=['stock_code', 'year', 'quarter'], 
                                 columns='category_code', 
                                 values='data', 
                                 aggfunc='sum')
    
    results = []

    # Iterate through the pivot table to calculate the new ratio
    for index, row in pivot_df.iterrows():
        stock_code, year, quarter = index
        row_index = set(row.index)
        
        for ratio, inputs in ratio_function.items():
            input_values = []

            for input_name in inputs:
                if isinstance(input_name, list):  # For cases like permanent capital (sum of multiple values)
                # Sum multiple values if input_name is a list
                    value_sum = sum([row[i] for i in input_name if i in row_index])
                    input_values.append(value_sum)
                else:
                    input_values.append(row[input_name] if input_name in row_index else None)
            
            # Check if all required data is available
            if None not in input_values:
                # Call the corresponding function to calculate the ratio
                ratio_value = globals()[ratio](*input_values)
                results.append({
                    'stock_code': stock_code,
                    'year': year,
                    'quarter': quarter,
                    'ratio_code': ratio,
                    'data': ratio_value
                })

    return pd.DataFrame(results)


def get_previous_year_q0_value(pivot_df, stock_code, year, category_code):
    try:
        return pivot_df.loc[(stock_code, year - 1, 0), category_code]
    except KeyError:
        return None


#=====================#
# Financial Structure #
#=====================#

# Ratio calculation functions
def equity_ratio(equity, total_assets):
    return equity / total_assets if total_assets else None

def long_term_asset_self_financing_ratio(permanent_capital, long_term_assets):
    return permanent_capital / long_term_assets if long_term_assets else None

def fixed_asset_self_financing_ratio(permanent_capital, fixed_assets):
    return permanent_capital / fixed_assets if fixed_assets else None

def general_solvency_ratio(total_assets, total_liabilities):
    return total_assets / total_liabilities if total_liabilities else None

def return_on_investment(net_income, total_investment):
    return net_income / total_investment if total_investment else None

def ROIC(NOPAT, invested_capital):
    return NOPAT / invested_capital if invested_capital else None

def return_on_long_term_capital(EBIT, average_long_term_capital):
    return EBIT / average_long_term_capital if average_long_term_capital else None

def basic_earning_power(EBIT, average_total_assets):
    return EBIT / average_total_assets if average_total_assets else None

def debt_to_assets_ratio(total_liabilities, total_assets):
    return total_liabilities / total_assets if total_assets else None

def debt_to_equity_ratio(total_liabilities, equity):
    return total_liabilities / equity if equity else None

def short_term_debt_to_assets_ratio(short_term_liabilities, total_assets):
    return short_term_liabilities / total_assets if total_assets else None

def interest_coverage_ratio(EBIT, interest_expense):
    return EBIT / interest_expense if interest_expense else None

def long_term_debt_to_equity_ratio(long_term_liabilities, equity):
    return long_term_liabilities / equity if equity else None

def short_term_debt_to_equity_ratio(short_term_liabilities, equity):
    return short_term_liabilities / equity if equity else None

def get_financial_structure_ratios(data_df, func_dict):

    return __get_financial_ratio(data_df, func_dict)



#===================#
#     Liquidity     #
#===================#


# Ratio calculation functions
def receivables_to_payables_ratio(accounts_receivable, total_liabilities):
    return accounts_receivable / total_liabilities if total_liabilities else None

def receivables_to_total_assets_ratio(accounts_receivable, total_assets):
    return accounts_receivable / total_assets if total_assets else None

def debt_to_total_capital_ratio(total_liabilities, total_capital):
    return total_liabilities / total_capital if total_capital else None

def receivables_to_sales_ratio(accounts_receivables, total_sales):
    return accounts_receivables / total_sales if total_sales else None

def allowance_for_doubtful_accounts_ratio(allowance_for_doubtful_accounts, accounts_receivables):
    return allowance_for_doubtful_accounts / accounts_receivables if accounts_receivables else None

def allowance_for_loan_customers_ratio(allowance_for_loan_customers, loan_to_customers):
    return allowance_for_loan_customers / loan_to_customers if loan_to_customers else None

def asset_to_debt_ratio(total_assets, total_liabilities):
    return total_assets / total_liabilities if total_liabilities else None

def current_ratio(current_assets, current_liabilities):
    return current_assets / current_liabilities if current_liabilities else None

def quick_ratio(current_assets, inventory, current_liabilities):
    return (current_assets - inventory) / current_liabilities if current_liabilities else None

def cash_ratio(cash_and_cash_equivalents, current_liabilities):
    return cash_and_cash_equivalents / current_liabilities if current_liabilities else None

def long_term_debt_coverage_ratio(non_current_assets, non_current_liabilities):
    return non_current_assets / non_current_liabilities if non_current_liabilities else None

def debt_to_equity_ratio(total_liabilities, total_equity):
    return total_liabilities / total_equity if total_equity else None

def long_term_debt_to_equity_capital_ratio(non_current_liabilities, equity):
    return non_current_liabilities / equity if equity else None

def time_interest_earned(EBIT, interest_expense):
    return EBIT / interest_expense if interest_expense else None

def debt_to_tangible_net_worth_ratio(total_liabilities, equity, intangible_assets):
    return total_liabilities / (equity - intangible_assets) if (equity - intangible_assets) else None


def get_liquidity_ratios(data_df, func_dict):
    return __get_financial_ratio(data_df, func_dict)

    
#===================#
#   Financial Risk  #
#===================#

    
# Financial Leverage = total_liabilities(BS_300) / total_lia_and_equity (BS_440)
def financial_leverage(total_liabilities, total_lia_and_equity):
    return total_liabilities / total_lia_and_equity if total_lia_and_equity else None

# Allowance for Doubtful Accounts to Total Assets Ratio = allowance_for_doubtful_accounts(BS_137+BS_219) / total_assets (BS_270)
def allowance_for_doubtful_accounts_to_total_assets_ratio(allowance_for_doubtful_accounts, total_assets):
    return allowance_for_doubtful_accounts / total_assets if total_assets else None

# Permanent Financing Ratio (Hệ số tài trợ thường xuyên) = permanent_capital(BS_400 + BS_330) / total_lia_and_equity (BS_440)
def permanent_financing_ratio(permanent_capital, total_lia_and_equity):
    return permanent_capital / total_lia_and_equity if total_lia_and_equity else None

def get_financial_risk_ratio(data_df, func_dict):
    return __get_financial_ratio(data_df, func_dict)


#===================#
#    Income ratio   #
#===================#

def financial_income_to_net_revenue_ratio(financial_income, net_revenue):
    return financial_income / net_revenue if net_revenue else None

def get_income_ratios(data_df, func_dict):
    return __get_financial_ratio(data_df, func_dict)


#===================#
#  Profitability    #
#===================#

# Ratio calculation functions
def return_on_assets(net_income, total_assets):
    return net_income / total_assets if total_assets else None

def return_on_fixed_assets(net_income, average_fixed_assets):
    return net_income / average_fixed_assets if average_fixed_assets else None

def return_on_long_term_operating_assets(net_income, average_long_term_operating_assets):
    return net_income / average_long_term_operating_assets if average_long_term_operating_assets else None

def Basic_Earning_Power_Ratio(EBIT, total_assets):
    return EBIT / total_assets if total_assets else None

def Return_on_equity(net_income, equity):
    return net_income / equity if equity else None

# def return_on_common_equity(net_income, preferred_dividends, average_common_equity):
#     return (net_income - preferred_dividends) / average_common_equity if average_common_equity else None

def profitability_of_cost_of_goods_sold(net_income_from_operating, COGS):
    return net_income_from_operating / COGS if COGS else None

def price_spread_ratio(gross_profit, COGS):
    return gross_profit / COGS if COGS else None

def profitability_of_operating_expenses(net_income_from_operating, total_operating_expenses):
    return net_income_from_operating / total_operating_expenses if total_operating_expenses else None

def Return_on_sales(net_income, net_sales):
    return net_income / net_sales if net_sales else None

def operating_profit_margin(net_profit_from_operating, net_sales, bank_params = None):
    # Poor code design, but it's the only way to pass the bank_params to the function
    if bank_params is not None:
        profit_from_operating, operating_expense, net_sales = net_profit_from_operating, net_sales, bank_params
        return (profit_from_operating - operating_expense)/ net_sales if net_sales else None
    
    return net_profit_from_operating / net_sales if net_sales else None

def gross_profit_margin(gross_profit, net_sales):
    return gross_profit / net_sales if net_sales else None

def get_profitability_ratios(data_df, func_dict):
    return __get_financial_ratio(data_df, func_dict)


#===================#
#   Cashflow ratio  #
#===================#

# Ratio calculation functions
def EBITDA(EBIT, depreciation_and_amortization):
    return EBIT + depreciation_and_amortization

def free_cash_flow(operating_net_cash_flow, capital_expenditures, dividends_paid):
    return operating_net_cash_flow - capital_expenditures - dividends_paid

def free_cash_flow_to_operating_cash_flow_ratio(free_cash_flow, operating_net_cash_flow):
    return free_cash_flow / operating_net_cash_flow if operating_net_cash_flow else None

def cash_debt_coverage_ratio(operating_net_cash_flow, avg_total_liabilities):
    return operating_net_cash_flow / avg_total_liabilities if avg_total_liabilities else None

def cash_interest_coverage(operating_net_cash_flow, interest_expense):
    return (operating_net_cash_flow + interest_expense) / interest_expense if interest_expense else None

def cash_return_on_assets(operating_net_cash_flow, avg_total_assets):
    return operating_net_cash_flow / avg_total_assets if avg_total_assets else None

def cash_return_on_fixed_assets(operating_net_cash_flow, avg_fixed_assets):
    return operating_net_cash_flow / avg_fixed_assets if avg_fixed_assets else None

def CFO_to_total_equity(operating_net_cash_flow, avg_total_equity):
    return operating_net_cash_flow / avg_total_equity if avg_total_equity else None

def cash_flow_from_sales_to_sales(operating_net_cash_flow, net_sales):
    return operating_net_cash_flow / net_sales if net_sales else None

def cash_flow_margin(operating_net_cash_flow, total_revenue):
    return operating_net_cash_flow / total_revenue if total_revenue else None

def earning_quality_ratio(operating_net_cash_flow, net_income):
    return operating_net_cash_flow / net_income if net_income else None

def net_interest_margin(net_interest_income, avg_earning_assets):
    return net_interest_income / avg_earning_assets if avg_earning_assets else None

def get_cashflow_ratios(data_df, func_dict):
    pivot_df_6 = data_df.pivot_table(index=['stock_code', 'year', 'quarter'], 
                                 columns='category_code', 
                                 values='data', 
                                 aggfunc='sum')
    
    cash_flow_results_6 = []

    # Iterate through the pivot table to calculate the cash flow ratios
    for index, row in pivot_df_6.iterrows():
        stock_code, year, quarter = index
        
        for ratio, inputs in func_dict.items():
            input_values = []
            for input_name in inputs:
                if isinstance(input_name, list):  # Sum for cases like capital_expenditures or total_revenue
                    value_sum = sum([row[i] for i in input_name if i in row.index])
                    input_values.append(value_sum)
                else:
                    if input_name in ['BS_300', 'BS_270', 'BS_220', 'BS_400']:  
                        prev_q0_value = get_previous_year_q0_value(pivot_df_6, stock_code, year, input_name)
                        current_value = row[input_name] if input_name in row.index else None
                        if current_value is not None and prev_q0_value is not None:
                            avg_value = (current_value + prev_q0_value) / 2
                            input_values.append(avg_value)
                        else:
                            input_values.append(None)
                    else:
                        input_values.append(row[input_name] if input_name in row.index else None)
            
            # Check if all required data is available
            if None not in input_values:
                # Call the corresponding function to calculate the ratio
                ratio_value = globals()[ratio](*input_values)
                cash_flow_results_6.append({
                    'stock_code': stock_code,
                    'year': year,
                    'quarter': quarter,
                    'ratio_code': ratio,
                    'data': ratio_value
                })

    # Convert the results to a DataFrame
    return pd.DataFrame(cash_flow_results_6)


#===================#
#   Main Function   #
#===================#

def get_constant_values(type_):
    if type_ == 'non_bank':
        return {
            'financial_structure': const.FINANCIAL_STRUCTURE_RATIO_FUNCTIONS,
            'liquidity': const.LIQUIDITY_RATIO_FUNCTIONS,
            'financial_risk': const.FINANCIAL_RATIO_FUNCTIONS,
            'income': const.INCOME_RATIO_FUNCTIONS,
            'profitability': const.PROFITABILITY_RATIO_FUNCTIONS,
            'cashflow': const.CASHFLOW_RATIO_FUNCTIONS
        }
        
    elif type_ == 'bank':
        return {
            'financial_structure': const.BANK_FINANCIAL_STRUCTURE_RATIO_FUNCTIONS,
            'liquidity': const.BANK_LIQUIDITY_RATIO_FUNCTIONS,
            'financial_risk': const.BANK_FINANCIAL_RATIO_FUNCTIONS,
            'income': const.BANK_INCOME_RATIO_FUNCTIONS,
            'profitability': const.BANK_PROFITABILITY_RATIO_FUNCTIONS,
            'cashflow': const.BANK_CASHFLOW_RATIO_FUNCTIONS
        }
    else:
        raise ValueError(f"Invalid type: {type_}")

def get_financial_ratios(data_df, type_ = 'non_bank'):
    
    constant = get_constant_values(type_)
    
    df_financial_structure = get_financial_structure_ratios(data_df, constant['financial_structure'])
    df_liquidity = get_liquidity_ratios(data_df, constant['liquidity'])
    df_financial_risk = get_financial_risk_ratio(data_df, constant['financial_risk'])
    df_income = get_income_ratios(data_df, constant['income'])
    df_profitability = get_profitability_ratios(data_df, constant['profitability'])
    df_cashflow = get_cashflow_ratios(data_df, constant['cashflow'])
    
    df = pd.concat([df_financial_structure, df_liquidity, df_financial_risk, df_income, df_profitability, df_cashflow], ignore_index=True)
    
    # Map ratio_code to ratio_name
    df.rename(columns={'ratio_code': 'function_name'}, inplace=True)

    map_df = pd.read_csv(os.path.join(current_path ,'../csv/map_ratio_code.csv'))
    map_df['function_name'] = map_df['function_name'].str.strip()
    df = pd.merge(df, map_df, on='function_name', how='left')
    df.drop_duplicates(inplace=True)
    
    # # Find the intersection (inner join)
    # set1 = set(df['ratio_mapping'])
    # set2 = set(ratio_df['ratio_mapping'])
    # intersection = set1.intersection(set2)

    # # Perform the outer join excluding the intersection
    # outer_join_excluding_inner = (set1.union(set2)) - intersection
    # assert len(outer_join_excluding_inner) == 0, f"Missing mapping for ratio: {outer_join_excluding_inner}"
    
    return df
    
    
if __name__ == '__main__':
    print("Test financial ratios")
    
    dfs = None
    types = ['non_bank', 'bank']
    
    for type_ in types:
        print(f"Processing {type_} data")
        data_df = pd.read_csv(os.path.join(current_path, f'../csv/{type_}_financial_report_v2_1.csv'))
        df = get_financial_ratios(data_df, type_)
        if dfs is None:
            dfs = df
        else:
            dfs = pd.concat([dfs, df], ignore_index=True)
    
    assert dfs['ratio_code'].isna().sum()==0 , "Null value in ratio_code"
    
    dfs.drop_duplicates(inplace=True)
    dfs.to_csv(os.path.join(current_path, '../csv/financial_ratio.csv'), index=False)
    
    ratio = dfs['ratio_code'].unique()
    
    for r in ratio[:5]:
        print(r)
        print(dfs[(dfs['ratio_code'] == r)&(dfs['quarter'] == 0)&(dfs['year'] == 2022)].head(5))
        print('========================================')