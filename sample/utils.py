import pandas as pd
from setup_db import DBHUB

def read_file_without_comments(file_path, start=["#", "//"]):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            if not any([line.startswith(s) for s in start]):
                new_lines.append(line)
        return '\n'.join(new_lines)
    
def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()
    
def edit_distance(str1, str2):
    # Initialize a matrix to store distances
    m = len(str1)
    n = len(str2)
    
    # Create a table to store results of subproblems
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill dp[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):
            
            # If the first string is empty, insert all characters of the second string
            if i == 0:
                dp[i][j] = j    
            
            # If the second string is empty, remove all characters of the first string
            elif j == 0:
                dp[i][j] = i    
            
            # If the last characters are the same, ignore it and recur for the remaining strings
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            
            # If the last character is different, consider all possibilities and find the minimum
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # Remove
                                   dp[i][j-1],    # Insert
                                   dp[i-1][j-1])  # Replace
    
    return dp[m][n]

def edit_distance_score(str1, str2):
    return 1 - edit_distance(str1, str2) / max(len(str1), len(str2))
    
    
def df_to_markdown(df):
    markdown = df.to_markdown(index=False)
    return markdown


def company_name_to_stock_code(db : DBHUB, names, method = 'similarity') -> pd.DataFrame:
    """
    Get the stock code based on the company name
    """
    if not isinstance(names, list):
        names = [names]
    
    if method == 'similarity': # Using similarity search
        return db.return_company_info(names)
    
    else: # Using rational DB
        dfs = []
        query = "SELECT * FROM company WHERE company_name LIKE '%{name}%'"
        
        if method == 'bm25-ts':
            query = "SELECT stock_code, company_name FROM company_info WHERE to_tsvector('english', company_name) @@ to_tsquery('{name}');"
        
        elif 'bm25' in method:
            pass # Using paradeDB
        
        else:
            raise ValueError("Method not supported")  
        
        for name in names:
            
            # Require translate company name in Vietnamese to English
            name = name # translate(name, 'vi', 'en')
            query = query.format(name=name)
            result = db.query(query, return_type='dataframe')
            
            dfs.append(result)
            
        if len(dfs) > 0:
            result = pd.concat(dfs)
        else:
            result = pd.DataFrame(columns=['stock_code', 'company_name'])
        return result

    
def get_company_detail_from_df(dfs, db: DBHUB, method = 'similarity') -> pd.DataFrame:
    stock_code = set()
    if not isinstance(dfs, list):
        dfs = [dfs]
    
    for df in dfs:
        for col in df.columns:
            if col == 'stock_code':
                stock_code.update(df[col].tolist())
            if col == 'company_name':
                stock_code.update(company_name_to_stock_code(db, df[col].tolist(), method)['stock_code'].tolist())
            if col == 'invest_on':
                stock_code.update(company_name_to_stock_code(db, df[col].tolist(), method)['stock_code'].tolist())
            
    list_stock_code = list(stock_code)
    
    return company_name_to_stock_code(db, list_stock_code, method)
    
if __name__ == '__main__':
    print(read_file_without_comments('prompt/seek_database.txt'))