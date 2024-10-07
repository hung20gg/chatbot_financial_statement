def read_file_without_comments(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line for line in lines if not line.startswith('#') and not line.startswith('//')]
        return ''.join(lines)
    
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
    
if __name__ == '__main__':
    print(read_file_without_comments('prompt/seek_database.txt'))