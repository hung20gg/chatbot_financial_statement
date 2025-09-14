import sys 
import os
current_path = os.path.dirname(__file__)
sys.path.append(current_path)


import const

import pandas as pd 
import numpy as np 
from tqdm import tqdm


import time

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


def get_previous_year_q0_value(pivot_df, stock_code, year, category_code, quarter=0):
    try:
        return pivot_df.loc[(stock_code, year - 1, quarter), category_code]
    except KeyError:
        return None

def get_pre_calculated_ratio(year, ratio_code, ratios_df, quarter=0):
    try:
        return ratios_df.loc[
                              (ratios_df['year'] == year) & 
                              (ratios_df['quarter'] == quarter) & 
                              (ratios_df['ratio_code'] == ratio_code), 
                              'data'].values[0]
    except (IndexError, KeyError):
        return None

def __get_yoy_ratios(data_df, ratios_df_1=None, ratios_df_6=None, ratio_mapping=None):
    """
    Computes the Year-over-Year (YoY) ratio dynamically for each quarter (Q1, Q2, Q3, Q4)
    and for annual data (Q0).
    """
    results = []
    
    # Pivot the data for easier lookup
    pivot_df = data_df.pivot_table(
        index=['stock_code', 'year', 'quarter'],
        columns='category_code',
        values='data',
        aggfunc='sum'
    )

    # Iterate through the pivoted data
    for (stock_code, year, quarter), row in pivot_df.iterrows():
        
        for ratio_name, category_code in ratio_mapping.items():
            try:
                if isinstance(category_code, list):
                    current_year_value = sum(row.get(code, 0) for code in category_code)
                    previous_year_value = sum(
                        pivot_df.loc[(stock_code, year - 1, quarter), code]
                        if (stock_code, year - 1, quarter) in pivot_df.index and code in pivot_df.columns
                        else 0 for code in category_code
                    )
                elif category_code in ['EBIT']:
                    current_year_value = get_pre_calculated_ratio(year, category_code, ratios_df_1, quarter)
                    previous_year_value = get_pre_calculated_ratio(year - 1, category_code, ratios_df_1, quarter)
                elif category_code in ['EBITDA']:
                    current_year_value = get_pre_calculated_ratio(year, category_code, ratios_df_6, quarter)
                    previous_year_value = get_pre_calculated_ratio(year - 1, category_code, ratios_df_6, quarter)
                else:
                    current_year_value = row.get(category_code, 0)
                    previous_year_value = pivot_df.loc[
                        (stock_code, year - 1, quarter), category_code
                    ] if (stock_code, year - 1, quarter) in pivot_df.index else 0
                
                
                # Calculate YoY growth
                yoy_value = None
                if previous_year_value == 0 or previous_year_value is None:
                    yoy_value = None  # Avoid division by zero
                else:
                    yoy_value = (current_year_value - previous_year_value) / previous_year_value

                # Store results
                results.append({
                    'stock_code': stock_code,
                    'year': year,
                    'quarter': quarter,
                    'ratio_code': ratio_name,
                    'data': yoy_value
                })


            except Exception as e:
                print(f"Error calculating YoY ratio {ratio_name} for {stock_code}, Q{quarter}, {year}: {e}")

    return pd.DataFrame(results)

def __get_qoq_ratios(data_df, ratios_df_1=None, ratios_df_6=None, ratio_mapping=None):
    """
    Computes the Quarter-on-Quarter (QoQ) ratio dynamically for each quarter (Q1, Q2, Q3, Q4).
    For Q1, compares it with Q4 of the previous year, while for other quarters, compares with the previous quarter.
    Excludes the calculation for quarter 0 since it represents annual data.
    """

    results = []

    pivot_df = data_df.pivot_table(
        index=['stock_code', 'year', 'quarter'],
        columns='category_code',
        values='data',
        aggfunc='sum'
    )

    for (stock_code, year, quarter), row in pivot_df.iterrows():
        
        if quarter == 0:
            continue 

        for ratio_name, category_code in ratio_mapping.items():
            try:
                if quarter == 1:
                    # Special case: For Q1, compare with Q4 of the previous year
                    if isinstance(category_code, list):
                        # Handle sums of multiple category codes
                        current_quarter_value = sum(row.get(code, 0) for code in category_code)
                        previous_quarter_value = sum(
                            pivot_df.loc[(stock_code, year - 1, 4), code]
                            if (stock_code, year - 1, 4) in pivot_df.index and code in pivot_df.columns
                            else 0 for code in category_code
                        )
                    elif category_code in ['EBIT']:
                        current_quarter_value = get_pre_calculated_ratio(year, category_code, ratios_df_1, quarter)
                        previous_quarter_value = get_pre_calculated_ratio(year - 1, category_code, ratios_df_1, 4)
                    elif category_code in ['EBITDA']:
                        current_quarter_value = get_pre_calculated_ratio(year, category_code, ratios_df_6, quarter)
                        previous_quarter_value = get_pre_calculated_ratio(year - 1, category_code, ratios_df_6, 4)
                    else:
                        # Standard case: Fetch current and previous quarter values for the same category
                        current_quarter_value = row.get(category_code, 0)
                        previous_quarter_value = pivot_df.loc[
                            (stock_code, year - 1, 4), category_code
                        ] if (stock_code, year - 1, 4) in pivot_df.index else 0
                else:
                    # Standard case: For other quarters, compare with the previous quarter
                    if isinstance(category_code, list):
                        current_quarter_value = sum(row.get(code, 0) for code in category_code)
                        previous_quarter_value = sum(
                            pivot_df.loc[(stock_code, year, quarter - 1), code]
                            if (stock_code, year, quarter - 1) in pivot_df.index and code in pivot_df.columns
                            else 0 for code in category_code
                        )
                    elif category_code in ['EBIT']:
                        current_quarter_value = get_pre_calculated_ratio(year, category_code, ratios_df_1, quarter)
                        previous_quarter_value = get_pre_calculated_ratio(year, category_code, ratios_df_1, quarter - 1)
                    elif category_code in ['EBITDA']:
                        current_quarter_value = get_pre_calculated_ratio(year, category_code, ratios_df_6, quarter)
                        previous_quarter_value = get_pre_calculated_ratio(year, category_code, ratios_df_6, quarter - 1)
                    else:
                        current_quarter_value = row.get(category_code, 0)
                        previous_quarter_value = pivot_df.loc[
                            (stock_code, year, quarter - 1), category_code
                        ] if (stock_code, year, quarter - 1) in pivot_df.index else 0

                # Calculate QoQ growth
                qoq_value = None
                if previous_quarter_value == 0 or previous_quarter_value is None:
                    qoq_value = None 
                else:
                    qoq_value = (current_quarter_value - previous_quarter_value) / previous_quarter_value

                results.append({
                    'stock_code': stock_code,
                    'year': year,
                    'quarter': quarter,
                    'ratio_code': ratio_name,
                    'data': qoq_value
                })

            except Exception as e:
                print(f"Error calculating QoQ ratio {ratio_name} for {stock_code}, Q{quarter}, {year}: {e}")

    return pd.DataFrame(results)


def __get_average_data(data_df, category_code):
    if isinstance(category_code, str):
        category_code = [category_code]

    category_code = np.array(category_code)

    # Filter and prepare data
   
   # Sum the data of the given category code if it is a list
    selected_data = (
        data_df[data_df['category_code'].isin(category_code)]
        .sort_values(by=['stock_code', 'year', 'quarter'])
    )
    if len(category_code) > 1:
        df = selected_data.groupby(['stock_code', 'year', 'quarter'])['data'].sum().reset_index()

    else:
        df = selected_data

    df['prev'] = df['data'].shift(1)

    df['average_data'] = (df['data'] + df['prev'].fillna(0)) / 2
    return df[['stock_code', 'year', 'quarter', 'average_data']]

    # Calculate the average value
    


def get_average_data(data_df, category_code, duration: str = 'all'):
    """
    Calculate the average value of a given category code for the given quarter.
    """
    


    is_yearly = False
    is_quarterly = False

    if duration == 'all':
        is_yearly = True
        is_quarterly = True
    elif duration == 'yearly':
        is_yearly = True
    elif duration == 'quarterly':
        is_quarterly = True

    dfs = []

    if is_yearly:

        data_yearly = data_df[data_df['quarter'] == 0]
        dfs.append(__get_average_data(data_yearly, category_code))
    
    if is_quarterly:
        
        data_quarterly = data_df[data_df['quarter'] != 0]
        dfs.append(__get_average_data(data_quarterly, category_code))

    return pd.concat(dfs)


def __single_cost_of_fund(data_df, cost, funds):
    # Filter and prepare inventory and COGS data
    df_cost = data_df[data_df['category_code'] == cost]

    # Calculate the average inventory
    avg_funds = get_average_data(data_df, funds, duration='all')

    # Merge the data
    merged_data = pd.merge(avg_funds, df_cost, on=['stock_code', 'year', 'quarter'], how='inner')

    merged_data['data'] = - merged_data['data'] / merged_data['average_data'].replace(0, np.nan)


    # Set the ratio code and select relevant columns
    merged_data = merged_data.assign(ratio_code='cost_of_fund').drop(columns=['average_data', 'category_code'])

    return merged_data


def __single_inventory_turnover_ratio(data_df, cost_of_goods_sold, inventory):
    # Filter and prepare inventory and COGS data
    cogs = data_df[data_df['category_code'] == cost_of_goods_sold]

    # Calculate the average inventory
    avg_inventory = get_average_data(data_df, inventory, duration='all')

    # Merge the data
    merged_data = pd.merge(avg_inventory, cogs, on=['stock_code', 'year', 'quarter'], how='inner')

    merged_data['data'] = merged_data['data'] / merged_data['average_data'].replace(0, np.nan)


    # Set the ratio code and select relevant columns
    merged_data = merged_data.assign(ratio_code='inventory_turnover_ratio').drop(columns=['average_data', 'category_code'])

    return merged_data


def router(function_name, *args, **kwargs):
    if function_name == 'get_yoy_ratio':
        return __get_yoy_ratios(*args, **kwargs)
    elif function_name == 'get_inventory_turnover_ratio':
        return __single_inventory_turnover_ratio(*args, **kwargs)
    elif function_name == 'get_cost_of_fund':
        return __single_cost_of_fund(*args, **kwargs)
    elif function_name == 'get_qoq_ratios':
        return __get_qoq_ratios(*args, **kwargs)
    else:
        return None

# This code run so long, gonna optimize it later @pphanhh
def get_yoy_ratios(data_df, type_, ratios_df_1=None, ratios_df_6=None, constant=None):
    """
    Calculate YoY ratios for the given dataset and function dictionary.
    Uses pre-calculated financial structure (ratios_df_1) and cash flow (ratios_df_6) ratios.
    """

    # Initialize results
    results = []
    inputs = []

    grouped_data = data_df.groupby('stock_code')

    is_ratio_df = False
    if ratios_df_1 is not None and ratios_df_6 is not None:
        is_ratio_df = True
        grouped_ratio_6 = ratios_df_6.groupby('stock_code')
        grouped_ratio_1 = ratios_df_1.groupby('stock_code')

    print("Before", data_df.shape)

    for symbol, df_symbol in grouped_data:

    # #     df_symbol = data_df[data_df['stock_code'] == symbol]
        
        inputs.append([
                    'get_yoy_ratio',
                    df_symbol,
                    grouped_ratio_1.get_group(symbol) if is_ratio_df else None,
                    grouped_ratio_6.get_group(symbol) if is_ratio_df else None,
                    constant
                ])
        
        
        if type_ == 'corp': # Add inventory turnover ratio for Corporation only
            inputs.append(
                [
                    'get_inventory_turnover_ratio',
                    df_symbol,
                    'BS_141',
                    'IS_011'
                ]
            )
        if type_ == 'bank':
            inputs.append(
                [
                    'get_cost_of_fund',
                    df_symbol,
                    'IS_002',
                    ['BS_310', 'BS_320', 'BS_330', 'BS_340', 'BS_360']
                ]
            )

    for args in inputs:
        results.append(router(*args))
    
    cleaneed_results = []
    for result in results:
        if result is not None:
            cleaneed_results.append(result)
    
    if len(cleaneed_results) == 0:
        return pd.DataFrame()
    return pd.concat(cleaneed_results)


def get_qoq_ratios(data_df, type_, ratios_df_1=None, ratios_df_6=None,  constant=None):
    """
    Calculate QoQ ratios for the given dataset using multi-processing.
    Uses pre-calculated financial structure (ratios_df_1) and cash flow (ratios_df_6) ratios.
    """

    results = []
    inputs = []

    print("Before", data_df.shape)
    all_category_code_required = set()
    for ratio_name, category_code in constant.items():
        if isinstance(category_code, list):
            all_category_code_required.update(category_code)
        else:
            all_category_code_required.add(category_code)
    all_category_code_required = np.array(list(all_category_code_required))
    print("Required category code", len(all_category_code_required))

    data_df = data_df[data_df['category_code'].isin(all_category_code_required)]
    print("After", data_df.shape)

    grouped_data = data_df.groupby('stock_code')

    is_ratio_df = False
    if ratios_df_1 is not None and ratios_df_6 is not None:
        is_ratio_df = True
        grouped_ratio_6 = ratios_df_6.groupby('stock_code')
        grouped_ratio_1 = ratios_df_1.groupby('stock_code')

    for symbol, df_symbol in grouped_data:

    # #     df_symbol = data_df[data_df['stock_code'] == symbol]
        
        inputs.append([
                    df_symbol[df_symbol['quarter'] != 0],
                    grouped_ratio_1.get_group(symbol) if is_ratio_df else None,
                    grouped_ratio_6.get_group(symbol) if is_ratio_df else None,
                    constant
                ])

    for args in inputs:
        results.append(__get_qoq_ratios(*args))

    cleaneed_results = []
    for result in results:
        if result is not None:
            cleaneed_results.append(result)
    
    if len(cleaneed_results) == 0:
        return pd.DataFrame()
    return pd.concat(cleaneed_results)




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


def cost_to_income_ratio(total_operating_expenses, net_sales):
    return total_operating_expenses / net_sales if net_sales else None

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
#     PE Ratio      #
#===================#

def earning_per_share(eps):
    return eps

def price_earning_ratio(price, eps):
    return price / eps if eps else None

def book_value_per_share(eps, net_income, equity):
    if eps == 0:
        return 0
    return equity/(net_income / eps) if net_income else None

def price_to_book_ratio(price, eps, net_income, equity):
    bvps = book_value_per_share(eps, net_income, equity)
    return price / bvps if bvps else None

def get_pe_ratios(data_df, func_dict):
    return __get_financial_ratio(data_df, func_dict)


#===================#
#    Date related   #
#===================#


def days_sales_outstanding(accounts_receivable, net_sales):
    return (accounts_receivable / net_sales) if net_sales else 0

def days_payable_outstanding(short, long, COGS):
    return (short + long) / COGS if COGS else 0

def days_inventory_outstanding(inventory, COGS):
    return (inventory / COGS) if COGS else 0

def cash_conversion_cycle(accounts_receivable, short, long, inventory, COGS, net_sales):
    DSO = days_sales_outstanding(accounts_receivable, net_sales)
    DPO = days_payable_outstanding(short, long, COGS)
    DIO = days_inventory_outstanding(inventory, COGS)
    return DSO + DIO - DPO

def get_date_related_ratios(data_df, func_dict):
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

def net_profit_margin(net_profit, net_sales):
    return net_profit / net_sales if net_sales else None


def return_on_assets_ttm(net_income_ttm, total_assets):
    return net_income_ttm / total_assets if total_assets else None

def return_on_equity_ttm(net_income_ttm, total_equity):
    return net_income_ttm / total_equity if total_equity else None

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
                    if type_ =='non_bank' and input_name in ['BS_220', ['BS_240','BS_210','BS_220','BS_230','BS_260']]:  
                        prev_q0_value = get_previous_year_q0_value(pivot_df_5,stock_code, year, input_name)
                        current_value = row[input_name] if input_name in row.index else None
                        if current_value is not None and prev_q0_value is not None:
                            avg_value = (current_value + prev_q0_value) / 2
                            input_values.append(avg_value)
                        else:
                            input_values.append(None)

                    elif type_ == 'bank' and input_name in [ ['BS_210','BS_220','BS_240'],'BS_220']: 
                        prev_q0_value = get_previous_year_q0_value(pivot_df_5, stock_code, year, input_name)
                        current_value = row[input_name] if input_name in row.index else None
                        if current_value is not None and prev_q0_value is not None:
                            avg_value = (current_value + prev_q0_value) / 2
                            input_values.append(avg_value)
                        else:
                            input_values.append(None)
                    
                    elif type_ == 'securities' and input_name in ['BS_220',['BS_211','BS_220','BS_230','BS_240','BS_250']]:
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
#  Average ratios   #
#===================#
def return_on_average_assets(net_income, avg_total_assets):
    return net_income / avg_total_assets if avg_total_assets else None

def return_on_average_equity(net_income, avg_total_equity):
    return net_income / avg_total_equity if avg_total_equity else None  
def return_on_average_sales(net_income, avg_total_sales):
    return net_income / avg_total_sales if avg_total_sales else None

def Total_Asset_Turnover(net_sales, avg_total_assets):
    return net_sales / avg_total_assets if avg_total_assets else None

def get_avg_ratios(data_df, func_dict, type_):
    pivot_df_7 = data_df.pivot_table(index=['stock_code', 'year', 'quarter'],
                                   columns='category_code',
                                   values='data',
                                   aggfunc='sum')
    
    avg_results_7 = []

    avg_accounts = {
        'bank': ['BS_300', 'BS_500', ['IS_010', 'IS_003', 'IS_004']],
        'non_bank': ['BS_270', 'BS_400', 'IS_010'],
        'securities': ['BS_270', 'BS_400', 'IS_020']
    }
    
    avg_denominator_ratios = [
        'return_on_average_assets', 
        'return_on_average_equity', 
        'return_on_average_sales', 
        'Total_Asset_Turnover'
    ]
    
    def get_value(pivot_df, stock_code, year, quarter, input_name):
        try:
            if isinstance(input_name, list):
                return sum([pivot_df.loc[(stock_code, year, quarter), i] 
                          for i in input_name if i in pivot_df.columns])
            else:
                return pivot_df.loc[(stock_code, year, quarter), input_name]
        except:
            return None

    def get_previous_year_q0_value(pivot_df, stock_code, year, input_name):
        return get_value(pivot_df, stock_code, year - 1, 0, input_name)
    
    def calculate_average(pivot_df, stock_code, year, quarter, input_name):
        current = get_value(pivot_df, stock_code, year, quarter, input_name)
        previous = get_previous_year_q0_value(pivot_df, stock_code, year, input_name)
        
        if current is not None and previous is not None:
            return (current + previous) / 2
        return None

    for index, row in pivot_df_7.iterrows():
        stock_code, year, quarter = index
   
        for ratio, inputs in func_dict.items():
            input_values = []

            if ratio in avg_denominator_ratios:
                # First input (numerator) - no averaging
                numerator = get_value(pivot_df_7, stock_code, year, quarter, inputs[0])
                input_values.append(numerator)
                
                # Second input (denominator) - needs averaging
                denominator = calculate_average(pivot_df_7, stock_code, year, quarter, inputs[1])
                input_values.append(denominator)
            else:

                for i, input_name in enumerate(inputs):

                    needs_averaging = False

                    if not isinstance(input_name, list) and input_name in avg_accounts[type_]:
                        needs_averaging = True
           
                    if (isinstance(input_name, list) and type_ == 'bank' and 
                        set(input_name) == set(['IS_010', 'IS_003', 'IS_004'])):
                        needs_averaging = True
                    
                    if needs_averaging:
                        value = calculate_average(pivot_df_7, stock_code, year, quarter, input_name)
                    else:
                        value = get_value(pivot_df_7, stock_code, year, quarter, input_name)
                    
                    input_values.append(value)
            
            if None not in input_values:
                ratio_value = globals()[ratio](*input_values)
                
                avg_results_7.append({
                    'stock_code': stock_code,
                    'year': year,
                    'quarter': quarter,
                    'ratio_code': ratio,
                    'data': ratio_value
                })
    
    return pd.DataFrame(avg_results_7)

#===================#
#  AVG TTM ratios   #
#===================#

def return_on_average_assets_ttm(net_income_ttm, avg_total_assets_ttm):
    return net_income_ttm / avg_total_assets_ttm if avg_total_assets_ttm else None

def return_on_average_equity_ttm(net_income_ttm, avg_total_equity_ttm):
    return net_income_ttm / avg_total_equity_ttm if avg_total_equity_ttm else None

def get_avg_ttm_ratios(data_df, func_dict):
   
    data_df = data_df[data_df['quarter'] != 0].copy()

    pivot_df_8 = data_df.pivot_table(
        index=['stock_code', 'year', 'quarter'],
        columns='category_code',
        values='data',
        aggfunc='sum'
    )

    results_8 = []

    net_income_ttm_code = func_dict['return_on_average_assets_ttm'][0]  # e.g., 'IS_060_TTM'
    total_assets_code = func_dict['return_on_average_assets_ttm'][1]    # e.g., 'BS_270'
    total_equity_code = func_dict['return_on_average_equity_ttm'][1]    # e.g., 'BS_400'
    
    pivot_df_8 = pivot_df_8.sort_index()
    def rolling_avg(series):
        if len(series) < 4:
            return None  # Ensure at least 4 periods exist
        return (series.iloc[0] + series.iloc[-1]) / 2

    pivot_df_8['avg_total_assets_ttm'] = (
        pivot_df_8.groupby(level=0)[total_assets_code]
        .rolling(window=4, min_periods=4)
        .apply(rolling_avg, raw=False)
        .reset_index(level=0, drop=True)
    )

    pivot_df_8['avg_total_equity_ttm'] = (
        pivot_df_8.groupby(level=0)[total_equity_code]
        .rolling(window=4, min_periods=4)
        .apply(rolling_avg, raw=False)
        .reset_index(level=0, drop=True)
    )

    for (stock_code, year, quarter), row in pivot_df_8.iterrows():
        # print(row.index)
        try:
            net_income_ttm = row.get(net_income_ttm_code, None)
            
            avg_total_assets_ttm = row.get('avg_total_assets_ttm', None)
            avg_total_equity_ttm = row.get('avg_total_equity_ttm', None)
       
            roa_ttm = return_on_average_assets_ttm(net_income_ttm, avg_total_assets_ttm) if net_income_ttm is not None else None
            roe_ttm = return_on_average_equity_ttm(net_income_ttm, avg_total_equity_ttm) if net_income_ttm is not None else None

            if roa_ttm is not None:
                results_8.append({'stock_code': stock_code, 'year': year, 'quarter': quarter, 'ratio_code':'return_on_average_assets_ttm', 'data': roa_ttm})
            if roe_ttm is not None:
                results_8.append({'stock_code': stock_code, 'year': year, 'quarter': quarter, 'ratio_code': 'return_on_average_equity_ttm', 'data': roe_ttm})

        except Exception as e:
            print(f"Error calculating TTM ratios for {stock_code}, Q{quarter}, {year}: {e}")

    return pd.DataFrame(results_8)

#===================#
#  Ratio from fiin  #
#===================#

def current_account_saving_account_ratio(total_deposit, demand_deposit, margin_deposit):
    return (demand_deposit + margin_deposit) / total_deposit if total_deposit else None

def bad_debt_ratio(total_loan, bad_debt):
    return bad_debt / total_loan if total_loan else None

def non_performing_loan_coverage_ratio(allowance, bad_debt):
    return -allowance / bad_debt if bad_debt else None

def get_financial_ratio_tm(data_df):
    return __get_financial_ratio(data_df, const.BANK_FIIN_RATIO_FUNCTIONS)
    






#===================#
#   Main Function   #
#===================#

def get_constant_values(type_):
    if type_ == 'corp':
        return {
            'financial_structure': const.FINANCIAL_STRUCTURE_RATIO_FUNCTIONS,
            'liquidity': const.LIQUIDITY_RATIO_FUNCTIONS,
            'financial_risk': const.FINANCIAL_RATIO_FUNCTIONS,
            'income': const.INCOME_RATIO_FUNCTIONS,
            'profitability': const.PROFITABILITY_RATIO_FUNCTIONS,
            'cashflow': const.CASHFLOW_RATIO_FUNCTIONS,
            'avg': const.CORP_AVG_RATIO_FUNCTIONS,
            'pe': const.PE_RATIO_FUNCTIONS,
            'yoy': const.YoY_RATIO_FUNCTIONS['non_bank'],
            'qoq': const.QoQ_RATIO_FUNCTIONS['non_bank'],
            'date': const.DATE_RELATED_FUNCTIONS,
            'avg_ttm': const.CORP_AVG_TTM_RATIO_FUNCTIONS
        }
        
    elif type_ == 'bank':
        return {
            'financial_structure': const.BANK_FINANCIAL_STRUCTURE_RATIO_FUNCTIONS,
            'liquidity': const.BANK_LIQUIDITY_RATIO_FUNCTIONS,
            'financial_risk': const.BANK_FINANCIAL_RATIO_FUNCTIONS,
            'income': const.BANK_INCOME_RATIO_FUNCTIONS,
            'profitability': const.BANK_PROFITABILITY_RATIO_FUNCTIONS,
            'cashflow': const.BANK_CASHFLOW_RATIO_FUNCTIONS,
            'avg': const.BANK_AVG_RATIO_FUNCTIONS,
            'pe': const.BANK_PE_RATIO_FUNCTIONS,
            'yoy': const.YoY_RATIO_FUNCTIONS['bank'],
            'qoq': const.QoQ_RATIO_FUNCTIONS['bank'],
            'date': const.BANK_DATE_RELATED_FUNCTIONS,
            'avg_ttm': const.BANK_AVG_TTM_RATIO_FUNCTIONS
        }
    
    elif type_ == 'securities':
        return {
            'financial_structure': const.SECURITIES_FINANCIAL_STRUCTURE_RATIO_FUNCTIONS,
            'liquidity': const.SECURITIES_LIQUIDITY_RATIO_FUNCTIONS,
            'financial_risk': const.SECURITIES_FINANCIAL_RATIO_FUNCTIONS,
            'income': const.SECURITIES_INCOME_RATIO_FUNCTIONS,
            'profitability': const.SECURITIES_PROFITABILITY_RATIO_FUNCTIONS,
            'cashflow': const.SECURITIES_CASHFLOW_RATIO_FUNCTIONS,
            'avg': const.SECURITIES_AVG_RATIO_FUNCTIONS,
            'pe': const.SECURITIES_PE_RATIO_FUNCTIONS,
            'yoy': const.YoY_RATIO_FUNCTIONS['securities'],
            'qoq': const.QoQ_RATIO_FUNCTIONS['securities'],
            'date': const.SECURITIES_DATE_RELATED_FUNCTIONS,
            'avg_ttm': const.SECURITIES_AVG_TTM_RATIO_FUNCTIONS
        }
        
    else:
        raise ValueError(f"Invalid type: {type_}")



# Helper function to execute the function with its arguments
def run_function(func, *args):
    return func(*args)


def remove_unwanted_ratios(constant, map_df):

    new_constant = dict()
    for key, value in constant.items():
        new_value = dict()
        for ratio, inputs in value.items():
            if ratio in map_df['function_name'].values:
                new_value[ratio] = inputs
        
        # Only add the key if there are ratios to calculate
        if new_value:
            new_constant[key] = new_value
    return new_constant


def get_financial_ratios(data_df, type_ = 'corp', including_explaination = True):


    map_df = pd.read_csv(os.path.join(current_path ,'../data/map_ratio_code.csv'))
    map_df['function_name'] = map_df['function_name'].str.strip()
    
    constant = get_constant_values(type_)
    constant = remove_unwanted_ratios(constant, map_df)

    dfs = []

    time_start = time.time()
    df_financial_structure = get_financial_structure_ratios(data_df, constant['financial_structure'])
    dfs.append(df_financial_structure)
    print("FS", df_financial_structure.columns, time.time() - time_start)
    
    time_start = time.time()
    df_liquidity = get_liquidity_ratios(data_df, constant['liquidity'])
    dfs.append(df_liquidity)
    print("LQ", df_liquidity.columns, time.time() - time_start)

    df_financial_risk = get_financial_risk_ratio(data_df, constant['financial_risk'])
    df_income = get_income_ratios(data_df, constant['income'])
    df_profitability = get_profitability_ratios(data_df, constant['profitability'],type_)
    df_cashflow = get_cashflow_ratios(data_df, constant['cashflow'], type_)

    dfs.extend([df_financial_risk, df_income, df_profitability, df_cashflow])

    time_start = time.time()
    df_avg = get_avg_ratios(data_df, constant['avg'], type_)
    print("AVG", df_avg.columns, time.time() - time_start)

    dfs.append(df_avg)

    try:
       
        if 'data' in constant['date']:
            time_start = time.time()
            df_date = get_date_related_ratios(data_df, constant['date'])
            print("Date", df_date.columns, time.time() - time_start)
            dfs.append(df_date)
    except Exception as e:
        print(f"Error calculating date-related ratios: {e}")
    
    try:
        if 'pe' in constant:
            time_start = time.time()
            df_pe = get_pe_ratios(data_df, constant['pe'])
            print("PE", df_pe.columns, time.time() - time_start)
            dfs.append(df_pe)
    except Exception as e:
        print(f"Error calculating Price ratios: {e}")
    
    try:
        if 'yoy' in constant:
            time_start = time.time()
            df_yoy = get_yoy_ratios(data_df, type_, ratios_df_1=df_financial_structure, ratios_df_6=df_cashflow, constant=constant['yoy'])
            print("YoY", df_yoy.columns, time.time() - time_start)
            dfs.append(df_yoy)
    except Exception as e:
        print(f"Error calculating YoY ratios: {e}")
    
    try:
        if 'avg_ttm' in constant:
            time_start = time.time()
            df_avg_ttm = get_avg_ttm_ratios(data_df, constant['avg_ttm'])
            print("AVG TTM", df_avg_ttm.columns, time.time() - time_start)
            dfs.append(df_avg_ttm)
    except Exception as e:
        print(f"Error calculating TTM ratios: {e}")

    try:
        if 'qoq' in constant:
            time_start = time.time()
            df_qoq = get_qoq_ratios(data_df, type_, ratios_df_1=df_financial_structure, ratios_df_6=df_cashflow, constant=constant['qoq'])
            print("QoQ", df_qoq.columns, time.time() - time_start)
            dfs.append(df_qoq)
    except Exception as e:
        print(f"Error calculating QoQ ratios: {e}")

    df = pd.concat(dfs, ignore_index=True)
    
    if type_ == 'bank' and including_explaination:
        df_tm = get_financial_ratio_tm(data_df)
        df = pd.concat([df, df_tm], ignore_index=True)
    
    # Map ratio_code to ratio_name
    df.rename(columns={'ratio_code': 'function_name'}, inplace=True)

    
    
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


def _get_date_added(data_df):
    quarter_to_month = {
        0: 12,  # Quarter 0 is December of the same year
        1: 3,
        2: 6,
        3: 9,
        4: 12
        }

    data_df['date_added'] = pd.to_datetime(data_df.apply(lambda row: f"{row['year']}-{quarter_to_month[row['quarter']]}-30", axis=1))
    return data_df


def modify_days_ratio(data_df):

    # These ratios are not multiplied by number of days in a quarter/year

    selected_ratio = ['DSO', 'DPO', 'DIO', 'CCC']
    index_selected =data_df['ratio_code'].isin(selected_ratio)

    index_annual_data = data_df['quarter'] == 0

    data_df.loc[index_selected & ~index_annual_data, 'data'] = data_df.loc[index_selected & index_annual_data, 'data'] * 90
    data_df.loc[index_selected & index_annual_data, 'data'] = data_df.loc[index_selected & index_annual_data, 'data'] * 365

    return data_df




def industry_ratios(data_df, metric = 'BS_400', top_n = 20, output_path = '../data/'):
    df_company = pd.read_csv(os.path.join(current_path, '../data/df_company_info.csv'))
    
    # Read the financial statement data to get top 10 industries
    df_fs = pd.read_parquet(os.path.join(current_path, output_path, 'financial_statement_v3.parquet'))

    # Add industry to the data
    df_fs = pd.merge(df_fs, df_company[['stock_code', 'industry']], on='stock_code', how='left')
    
    top_20_stocks = (
        df_fs[df_fs['category_code'] == metric]
        .groupby(['industry', 'year', 'quarter'], group_keys=False)
        .apply(lambda x: x.nlargest(top_n, 'data'))
        .reset_index(drop=True)
    )[['industry', 'year', 'quarter', 'stock_code']]

    # Inner Join of top 20 stocks with the financial statement data
    filtered_data = pd.merge(data_df, top_20_stocks, on=['year', 'quarter', 'stock_code'], how='left')
    print(filtered_data[filtered_data['ratio_code'] == 'BDR'].head(10))

    # Get the mean financial ratios for the top 20 stocks in each industry
    df_industry_ratios = filtered_data.groupby(['industry', 'year', 'quarter', 'ratio_code'])['data'].mean().reset_index()
    
    print(df_industry_ratios[(df_industry_ratios['ratio_code'] == 'BDR') & (df_industry_ratios['year'] == 2023)].head(10))

    assert data_df['ratio_code'].nunique() == df_industry_ratios['ratio_code'].nunique(), "Missing ratio code in industry_ratios"

    df_industry_ratios.rename(columns={'data': 'data_mean'}, inplace=True)
    
    df_industry_ratios = _get_date_added(df_industry_ratios)

    return df_industry_ratios


def calculate_index(version = 'v3', output_path: str = '../data/'):

    if version == 'v3':
        including_explaination = True
    else:
        including_explaination = False

    dfs = []
    types = ['corp',  'bank', 'securities']

    df_stock_price_quarter = pd.read_parquet(os.path.join(current_path, '../csv/stock_price_quarterly.parquet'))
    df_stock_price_quarter['category_code'] = 'Price'
    df_stock_price_quarter['data'] = df_stock_price_quarter['close']
    df_stock_price_quarter = df_stock_price_quarter[['stock_code', 'year', 'quarter', 'category_code', 'data']]

    # Iterate through the types
    for type_ in types:
        print(f"Processing {type_} data")

        # Read the financial statement data
        data_df = pd.read_parquet(os.path.join(current_path, f'../data/{version}/{type_}_financial_report.parquet'))
        
        # Get the stock price data based on type
        df_stock_price_type = df_stock_price_quarter[df_stock_price_quarter['stock_code'].isin(data_df['stock_code'].unique())]
        # Merge the data with the stock price data
        data_df = pd.concat([data_df, df_stock_price_type], ignore_index=True)

        # Get the explaination if the type is bank and INCLUDING_FIIN is True
        if including_explaination and type_ == 'bank':
            tm_df = pd.read_parquet(os.path.join(current_path, '../data/v3/bank_explaination.parquet'))
            data_df = pd.concat([data_df, tm_df], ignore_index=True)
        
        
        # Get the financial ratios
        df = get_financial_ratios(data_df[['stock_code', 'year', 'quarter', 'category_code', 'data']], type_, including_explaination =including_explaination)
        
        # Format datetime
        data_df['time_code'] = data_df['stock_code'] + data_df['year'].astype(str) + data_df['quarter'].astype(str)
        df['time_code'] = df['stock_code'] + df['year'].astype(str) + df['quarter'].astype(str)
        

        time_df = data_df[['time_code', 'date_added']]
        time_df = time_df.dropna().drop_duplicates().copy()
        # time_df.drop_duplicates(inplace=True)
        
        df = pd.merge(df, time_df, on='time_code', how='inner')
        df.drop(columns=['time_code'], inplace=True)
        
        dfs.append(df)


    # Concatenate the dataframes
    dfs = pd.concat(dfs, ignore_index=True)
    ttm_account = dfs.ratio_code.str.contains('TTM')
    dfs = dfs[~((dfs['quarter'] == 0) & ttm_account)]
    
    assert dfs['ratio_code'].isna().sum()==0 , "Null value in ratio_code"
    assert dfs['date_added'].isna().sum()==0 , "Null value in date_added"
    
    dfs.drop_duplicates(inplace=True)
    dfs.fillna(0, inplace=True)
    
    dfs = dfs[['stock_code', 'year', 'quarter', 'ratio_code', 'data', 'date_added']]
    
    dfs = modify_days_ratio(dfs)

    dfs.to_parquet(os.path.join(current_path, output_path ,'financial_ratio_v3.parquet'), index=False)
    

    # Get the industry ratios

    df_industry_ratios = industry_ratios(dfs, metric='BS_400', top_n=15, output_path=output_path)
    print(df_industry_ratios[(df_industry_ratios['ratio_code'] == 'BDR') & (df_industry_ratios['year'] == 2023)].head(10))
    df_industry_ratios.to_parquet(os.path.join(current_path, output_path, f'industry_ratio_{version}.parquet'), index=False)

    

    return dfs, df_industry_ratios

if __name__ == '__main__':
    
    start = time.time()
    print("Test financial ratios")
    dfs, df_industry_ratios = calculate_index(version='v3')

    end = time.time()

    ratio = dfs['ratio_code'].unique().tolist()
    
    for r in ['BDR', 'EPS', 'BVPS', 'PB', 'DSO', 'ROAATTM', 'GPGYoY', 'GPGQoQ']:
        print(r)
        print(dfs[(dfs['ratio_code'] == r)&(dfs['quarter'] == 0)&(dfs['year'] == 2022)].head(5))
        print('========================================')

    for r in ['ROATTM', 'GPGQoQ']:
        print(r)

        bank_code = ['BID', 'CTG', 'VCB', 'MBB', 'ACB', 'TCB', 'VPB', 'HDB', 'TPB', 'EIB']
        print(dfs[(dfs['ratio_code'] == r)&(dfs['quarter'] == 2)&(dfs['year'] == 2022)&(dfs['stock_code'].isin(bank_code))].head(10))
        print('========================================')

    print(f"Time: {end - start}")