import sys 
import os
current_path = os.path.dirname(__file__)
sys.path.append(current_path)


import const

import pandas as pd 
import numpy as np 


INCLUDING_FIIN = True

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
                    if input_name in row_index:
                        input_values.append(row[input_name])
            
            # Check if all required data is available
            if None not in input_values:
                # Call the corresponding function to calculate the ratio
                # print(ratio, input_values)
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

def get_pre_calculated_ratio(stock_code, year, ratio_code, ratios_df):
    try:
        return ratios_df.loc[(ratios_df['stock_code'] == stock_code) & 
                              (ratios_df['year'] == year) & 
                              (ratios_df['quarter'] == 0) & 
                              (ratios_df['ratio_code'] == ratio_code), 
                              'data'].values[0]
    except (IndexError, KeyError):
        return None

def get_yoy_ratios(data_df, type_, ratios_df_1=None, ratios_df_6=None):
    """
    Calculate YoY ratios for the given dataset and function dictionary.
    Uses pre-calculated financial structure (ratios_df_1) and cash flow (ratios_df_6) ratios.
    """
    pivot_df = data_df.pivot_table(
        index=['stock_code', 'year', 'quarter'],
        columns='category_code',
        values='data',
        aggfunc='sum'
    )

    # Get YoY ratio mappings from const.py
    ratio_mapping = const.YoY_RATIO_FUNCTIONS[type_]

    # Initialize results
    results = []

    # Iterate through the pivoted data
    for (stock_code, year, quarter), row in pivot_df.iterrows():
        if quarter != 0:  # Skip non-annual data
            continue

        for ratio_name, category_code in ratio_mapping.items():
            # Process YoY growth calculation
            try:
                if isinstance(category_code, list):
                    # Handle sums of multiple codes
                    current_year_value = sum(row.get(code, 0) for code in category_code)
                    previous_year_value = sum(
                        pivot_df.loc[(stock_code, year - 1, 0), code]
                        if (stock_code, year - 1, 0) in pivot_df.index and code in pivot_df.columns
                        else 0 for code in category_code
                    )
                elif category_code in ['EBIT']:
                    current_year_value = get_pre_calculated_ratio(stock_code, year, category_code, ratios_df_1)
                    previous_year_value = get_pre_calculated_ratio(stock_code, year - 1, category_code, ratios_df_1)
                elif category_code in ['EBITDA']:
                    current_year_value = get_pre_calculated_ratio(stock_code, year, category_code, ratios_df_6)
                    previous_year_value = get_pre_calculated_ratio(stock_code, year - 1, category_code, ratios_df_6)
                else:
                    current_year_value = row.get(category_code, None)
                    previous_year_value = pivot_df.loc[
                        (stock_code, year - 1, 0), category_code
                    ] if (stock_code, year - 1, 0) in pivot_df.index else None

                # Calculate YoY growth
                yoy_value = None
                if previous_year_value == 0 or previous_year_value is None:
                    yoy_value = None  # Avoid division by zero
                else:
                    yoy_value = (current_year_value - previous_year_value) / previous_year_value
                    results.append({
                        'stock_code': stock_code,
                        'year': year,
                        'quarter': quarter,
                        'ratio_code': ratio_name,
                        'data': yoy_value
                    })
            except Exception as e:
                # Handle missing or invalid data gracefully
                print(f"Error calculating YoY ratio {ratio_name} for {stock_code}: {e}")

    return pd.DataFrame(results)



#=====================#
# Financial Structure #
#=====================#

# Ratio calculation functions

def EBIT(income_before_tax, interest_expense = None):
    if interest_expense is None:
        return income_before_tax
    return income_before_tax + interest_expense

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
    if invested_capital is None:
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

def Basic_Earning_Power_Ratio(EBIT, avg_total_assets):
    return EBIT / avg_total_assets if avg_total_assets else None

def Return_on_equity(net_income, equity):
    return net_income / equity if equity else None

# def return_on_common_equity(net_income, preferred_dividends, average_common_equity):
#     return (net_income - preferred_dividends) / average_common_equity if average_common_equity else None

def profitability_of_cost_of_goods_sold(net_income_from_operating, COGS):
    return net_income_from_operating / COGS if COGS else None

def price_spread_ratio(gross_profit, COGS):
    return gross_profit / COGS if COGS else None

def profitability_of_operating_expenses(net_income_from_operating, total_operating_expenses, operating_expense = None):
    
    if operating_expense is None:
        return net_income_from_operating / total_operating_expenses if total_operating_expenses else None
    else:
        def profitability_of_operating_expenses2(profit_from_operating, operating_expense, total_operating_expenses):
            return  (profit_from_operating - operating_expense) / total_operating_expenses if total_operating_expenses else None
        return profitability_of_operating_expenses2(net_income_from_operating, operating_expense, total_operating_expenses)

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

def Total_Asset_Turnover(net_sales, avg_total_assets):
    return net_sales / avg_total_assets if avg_total_assets else None

def get_profitability_ratios(data_df, func_dict, type_):
    pivot_df_5 = data_df.pivot_table(index=['stock_code', 'year', 'quarter'], 
                                 columns='category_code', 
                                 values='data', 
                                 aggfunc='sum')
    profitability_results_5 = []

    # Iterate through the pivot table to calculate the new ratios
    for index, row in pivot_df_5.iterrows():
        stock_code, year, quarter = index
        
        for ratio, inputs in func_dict.items():
            input_values = []
            for input_name in inputs:
                if isinstance(input_name, list):  
                    value_sum = sum([row[i] for i in input_name if i in row.index])
                    input_values.append(value_sum)
                else:
                    if type_ =='non_bank' and input_name in ['BS_220', ['BS_240','BS_210','BS_220','BS_230','BS_260'], 'BS_270']:  
                        prev_q0_value = get_previous_year_q0_value(pivot_df_5,stock_code, year, input_name)
                        current_value = row[input_name] if input_name in row.index else None
                        if current_value is not None and prev_q0_value is not None:
                            avg_value = (current_value + prev_q0_value) / 2
                            input_values.append(avg_value)
                        else:
                            input_values.append(None)

                    elif type_ == 'bank' and input_name in ['BS_300', ['BS_210','BS_220','BS_240'],'BS_220']: 
                        prev_q0_value = get_previous_year_q0_value(pivot_df_5, stock_code, year, input_name)
                        current_value = row[input_name] if input_name in row.index else None
                        if current_value is not None and prev_q0_value is not None:
                            avg_value = (current_value + prev_q0_value) / 2
                            input_values.append(avg_value)
                        else:
                            input_values.append(None)
                    
                    elif type_ == 'securities' and input_name in ['BS_220', 'BS_270',['BS_211','BS_220','BS_230','BS_240','BS_250']]:
                        prev_q0_value = get_previous_year_q0_value(pivot_df_5, stock_code, year, input_name)
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
                profitability_results_5.append({
                    'stock_code': stock_code,
                    'year': year,
                    'quarter': quarter,
                    'ratio_code': ratio,
                    'data': ratio_value
                })

    # Convert the results to a DataFrame
    return pd.DataFrame(profitability_results_5)



#===================#
#   Cashflow ratio  #
#===================#

# Ratio calculation functions
def EBITDA(EBIT, depreciation_and_amortization = None):
    if depreciation_and_amortization is None:
        return EBIT
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

def get_cashflow_ratios(data_df, func_dict, type_):
    pivot_df_6 = data_df.pivot_table(index=['stock_code', 'year', 'quarter'], 
                                 columns='category_code', 
                                 values='data', 
                                 aggfunc='sum')
    
    cash_flow_results_6 = []

    # Iterate through the pivot table to calculate the cash flow ratios
    for index, row in pivot_df_6.iterrows():
        stock_code, year, quarter= index
        
        for ratio, inputs in func_dict.items():
            input_values = []
            for input_name in inputs:
                if isinstance(input_name, list):  # Sum for cases like capital_expenditures or total_revenue
                    value_sum = sum([row[i] for i in input_name if i in row.index])
                    input_values.append(value_sum)
                else:
                    if type_ == 'bank' and input_name in ['BS_400', 'BS_300', 'BS_220', 'BS_500'] :
                        prev_q0_value = get_previous_year_q0_value(pivot_df_6, stock_code, year, input_name)
                        current_value = row[input_name] if input_name in row.index else None
                        if current_value is not None and prev_q0_value is not None:
                            avg_value = (current_value + prev_q0_value) / 2
                            input_values.append(avg_value)
                        else:
                            input_values.append(None)
                            
                    elif type_ in ['non_bank', 'securities'] and input_name in ['BS_300', 'BS_270', 'BS_220', 'BS_400']: 
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
#  Ratio from fiin  #
#===================#

def current_account_saving_account_ratio(total_deposit, demand_deposit, margin_deposit):
    return (demand_deposit + margin_deposit) / total_deposit if total_deposit else None

def bad_debt_ratio(total_loan, bad_debt):
    return bad_debt / total_loan if total_loan else None

def get_financial_ratio_tm(data_df):
    return __get_financial_ratio(data_df, const.BANK_FIIN_RATIO_FUNCTIONS)
    

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
            'cashflow': const.CASHFLOW_RATIO_FUNCTIONS,
            'yoy': const.YoY_RATIO_FUNCTIONS['non_bank']
        }
        
    elif type_ == 'bank':
        return {
            'financial_structure': const.BANK_FINANCIAL_STRUCTURE_RATIO_FUNCTIONS,
            'liquidity': const.BANK_LIQUIDITY_RATIO_FUNCTIONS,
            'financial_risk': const.BANK_FINANCIAL_RATIO_FUNCTIONS,
            'income': const.BANK_INCOME_RATIO_FUNCTIONS,
            'profitability': const.BANK_PROFITABILITY_RATIO_FUNCTIONS,
            'cashflow': const.BANK_CASHFLOW_RATIO_FUNCTIONS,
            'yoy': const.YoY_RATIO_FUNCTIONS['bank']
        }
    
    elif type_ == 'securities':
        return {
            'financial_structure': const.SECURITIES_FINANCIAL_STRUCTURE_RATIO_FUNCTIONS,
            'liquidity': const.SECURITIES_LIQUIDITY_RATIO_FUNCTIONS,
            'financial_risk': const.SECURITIES_FINANCIAL_RATIO_FUNCTIONS,
            'income': const.SECURITIES_INCOME_RATIO_FUNCTIONS,
            'profitability': const.SECURITIES_PROFITABILITY_RATIO_FUNCTIONS,
            'cashflow': const.SECURITIES_CASHFLOW_RATIO_FUNCTIONS,
            'yoy': const.YoY_RATIO_FUNCTIONS['securities']
        }
        
    else:
        raise ValueError(f"Invalid type: {type_}")

def get_financial_ratios(data_df, type_ = 'non_bank'):
    
    constant = get_constant_values(type_)
    
    df_financial_structure = get_financial_structure_ratios(data_df, constant['financial_structure'])
    df_liquidity = get_liquidity_ratios(data_df, constant['liquidity'])
    df_financial_risk = get_financial_risk_ratio(data_df, constant['financial_risk'])
    df_income = get_income_ratios(data_df, constant['income'])
    df_profitability = get_profitability_ratios(data_df, constant['profitability'],type_)
    df_cashflow = get_cashflow_ratios(data_df, constant['cashflow'], type_)
    df_yoy = get_yoy_ratios(data_df, type_, ratios_df_1=df_financial_structure, ratios_df_6=df_cashflow)
    df = pd.concat([df_financial_structure, df_liquidity, df_financial_risk, df_income, df_profitability, df_cashflow, df_yoy], ignore_index=True)
    
    if type_ == 'bank' and INCLUDING_FIIN:
        df_tm = get_financial_ratio_tm(data_df)
        df = pd.concat([df, df_tm], ignore_index=True)
    
    # Map ratio_code to ratio_name
    df.rename(columns={'ratio_code': 'function_name'}, inplace=True)

    map_df = pd.read_csv(os.path.join(current_path ,'../csv/map_ratio_code.csv'))
    map_df['function_name'] = map_df['function_name'].str.strip()
    df = pd.merge(df, map_df, on='function_name', how='left')
    # Check for missing mappings
    missing_ratios = df[df['ratio_code'].isna()]
    if not missing_ratios.empty:
        print("Missing ratio codes for the following function names:")
        print(missing_ratios['function_name'].unique())
        raise ValueError("Update map_ratio_code.csv with the missing function names.")
    df.drop(columns=['function_name'], inplace=True)
    # print(df[df['ratio_code'].isna()]['function_name'].unique())
    
    df.drop_duplicates(inplace=True)
    
    return df


    
    
if __name__ == '__main__':
    
    
    
    print("Test financial ratios")
    
    dfs = None
    types = ['non_bank', 'bank', 'securities']
    
    for type_ in types:
        print(f"Processing {type_} data")
        data_df = pd.read_parquet(os.path.join(current_path, f'../csv/{type_}_financial_report_v3.parquet'))
        
        if INCLUDING_FIIN and type_ == 'bank':
            tm_df = pd.read_parquet(os.path.join(current_path, '../csv/bank_explaination_v3.parquet'))
            data_df = pd.concat([data_df, tm_df], ignore_index=True)
        
        df = get_financial_ratios(data_df[['stock_code', 'year', 'quarter', 'category_code', 'data']], type_)
        
        data_df['time_code'] = data_df['stock_code'] + data_df['year'].astype(str) + data_df['quarter'].astype(str)
        df['time_code'] = df['stock_code'] + df['year'].astype(str) + df['quarter'].astype(str)
        
        time_df = data_df[['time_code', 'date_added']].drop_duplicates()
        
        df = pd.merge(df, time_df, on='time_code', how='left')
        df.drop(columns=['time_code'], inplace=True)
        
        if dfs is None:
            dfs = df
        else:
            dfs = pd.concat([dfs, df], ignore_index=True)
    
    assert dfs['ratio_code'].isna().sum()==0 , "Null value in ratio_code"
    
    dfs.drop_duplicates(inplace=True)
    dfs.fillna(0, inplace=True)
    
    
    dfs.to_parquet(os.path.join(current_path, '../csv/financial_ratio_v3.parquet'), index=False)
    
    ratio = dfs['ratio_code'].unique()
    
    for r in ratio[:5]:
        print(r)
        print(dfs[(dfs['ratio_code'] == r)&(dfs['quarter'] == 0)&(dfs['year'] == 2022)].head(5))
        print('========================================')