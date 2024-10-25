import ratio_const
import pandas as pd 
import numpy as np 


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

def get_financial_structure_ratios(data_df):
    pivot_df = data_df.pivot_table(index=['stock_code', 'year', 'quarter'], 
                                           columns='category_code', 
                                           values='data', 
                                           aggfunc='sum')
    
    # Create a DataFrame to store the results
    results_1 = []

    # Iterate through pivoted data
    for index, row in pivot_df.iterrows():
        stock_code, year, quarter = index
        
        for ratio, inputs in ratio_const.FINANCIAL_STRUCTURE_RATIO_FUNCTIONS.items():
            # Fetch input data from the pivot table
            input_values = []
            for input_name in inputs:
                if isinstance(input_name, list):  # For cases like permanent capital (sum of multiple values)
                    # Sum multiple values if input_name is a list
                    value_sum = sum([row[i] for i in input_name if i in row.index])
                    input_values.append(value_sum)
                else:
                    input_values.append(row[input_name] if input_name in row.index else None)
            
            # Check if all required data is available
            if None not in input_values:
                # Call the corresponding function to calculate the ratio
                ratio_value = globals()[ratio](*input_values)
                results_1.append({
                    'stock_code': stock_code,
                    'year': year,
                    'quarter': quarter,
                    'ratio_code': ratio,
                    'data': ratio_value
                })

    # Convert results to DataFrame
    ratios_df_1 = pd.DataFrame(results_1)

    return ratios_df_1


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


def get_liquidity_ratios(data_df):
    results_2 = []
    pivot_df_2 = data_df.pivot_table(index=['stock_code', 'year', 'quarter'], 
                                           columns='category_code', 
                                           values='data', 
                                           aggfunc='sum')

    # Iterate through pivoted data
    for index, row in pivot_df_2.iterrows():
        stock_code, year, quarter = index
        
        for ratio, inputs in ratio_const.LIQUIDITY_RATIO_FUNCTIONS.items():
            input_values = []
            if ratio == 'current_ratio':
                # Subtract BS_151 from BS_100 for current ratio calculation
                current_assets = row[inputs[0][0]] - row[inputs[0][1]] if inputs[0][0] in row.index and inputs[0][1] in row.index else None
                input_values.append(current_assets)
                input_values.append(row[inputs[1]] if inputs[1] in row.index else None)
            elif ratio == 'quick_ratio':
                # Subtract BS_140 from BS_100 for quick ratio calculation
                current_assets = row[inputs[0][0]] - row[inputs[0][1]] if inputs[0][0] in row.index and inputs[0][1] in row.index else None
                inventory = row[inputs[1]] if inputs[1] in row.index else None
                current_liabilities = row[inputs[2]] if inputs[2] in row.index else None
                input_values.extend([current_assets, inventory, current_liabilities])
            else:
                # For all other ratios, sum values if needed or fetch directly
                for input_name in inputs:
                    if isinstance(input_name, list):
                        value_sum = sum([row[i] for i in input_name if i in row.index])
                        input_values.append(value_sum)
                    else:
                        input_values.append(row[input_name] if input_name in row.index else None)
            
            # Check if all required data is available
            if None not in input_values:
                # Call the corresponding function to calculate the ratio
                ratio_value = globals()[ratio](*input_values)
                results_2.append({
                    'stock_code': stock_code,
                    'year': year,
                    'quarter': quarter,
                    'ratio_code': ratio,
                    'data': ratio_value
                })

    # Convert results to DataFrame
    ratios_df_2 = pd.DataFrame(results_2)

    # Display or export the DataFrame
    ratios_df_2