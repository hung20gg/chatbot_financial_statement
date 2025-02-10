-- Table: company_info
CREATE TABLE company_info(
    stock_code VARCHAR(255) primary key, --The trading symbol.
    industry VARCHAR(255), --Current industry of company. 
    exchange VARCHAR(255), -- The market where the stock is listed (e.g., HOSE, HNX)
    stock_indices VARCHAR(255), -- The stock index it belongs to (e.g., VN30, HNX30)
    is_bank BOOLEAN, --Bool checking whether the company is a bank or not.
    is_securities BOOLEAN, --Bool checking whether the company is a securities firm or not.
    issue_share int --Number of share issued.
);

-- Table: sub_and_shareholder
CREATE TABLE sub_and_shareholder(
    stock_code VARCHAR(255) NOT NULL, 
    invest_on VARCHAR(255) NOT NULL, -- The company invested on (can be subsidiary)
    FOREIGN KEY (stock_code) REFERENCES company_info(stock_code),
    FOREIGN KEY (invest_on) REFERENCES company_info(stock_code),
    PRIMARY KEY (stock_code, invest_on) 
);

-- Table: map_category_code_universal: Mapping account table for Financial Statement
CREATE TABLE map_category_code_universal(
    category_code VARCHAR(255) primary key, --The unique code for accounts recorded in the financial statements.
    en_caption VARCHAR(255), --The Accounts (Caption) for the `category_code`.
);

-- Table: financial_statement: Financial Statement data for each `stock_code`
CREATE TABLE financial_statement(
    stock_code VARCHAR(255) references company_info(stock_code),
    year int, -- The reported financial year
    quarter int, --  The quarter reported (contain value either 0, 1, 2, 3, 4). If the value is 0, that mean the report is for annual report.
    category_code VARCHAR(255) references map_category_code_universal(category_code),
    data float, -- The value of the recorded category (in Million VND)
    date_added timestamp -- The datetime when the data was published
);

-- Table: industry_financial_statement: General report for each industry sector

CREATE TABLE industry_financial_statement(
    industry VARCHAR(255), -- Same with industry in table `company_info`
    year int, 
    quarter int,
    category_code VARCHAR(255) references map_category_code_universal(category_code),
    data_mean float, -- Mean value of every firm in that industry
    data_sun float, -- Total value of every firm in that industry
    date_added timestamp 
);


-- Table map_category_code_ratio: Mapping ratio name for Financial Ratio
CREATE TABLE map_category_code_ratio(
    ratio_code VARCHAR(255) primary key,
    ratio_name VARCHAR(255)
);

-- Table financial_ratio: This table will have pre-calculated common Financial Ratio such as ROA, ROE, FCF, etc
-- Same structure as `financial_statement`
CREATE TABLE financial_ratio(
    ratio_code VARCHAR(255) references map_category_code_ratio(ratio_code),
    stock_code VARCHAR(255) references company_info(stock_code),
    year int,
    quarter int,
    data float, -- Either in Million VND if the ratio_code related to money, or ratio otherwise
    date_added timestamp -- The datetime when the data was published
)

-- Table: industry_financial_ratio: General ratio for each industry sector
CREATE TABLE industry_financial_ratio(
    industry VARCHAR(255),
    ratio_code VARCHAR(255) references map_category_code_ratio(ratio_code),
    year int,
    quarter int,
    data_mean float, 
    date_added timestamp -- The datetime when the data was published
)

-- Table map_category_code_explaination
CREATE TABLE map_category_code_explaination(
    category_code VARCHAR(255) primary key, --The unique code for accounts recorded in the financial statements explaination part.
    en_caption VARCHAR(255), --The Accounts (Caption) for the `category_code`.
);

-- Table financial_statement_explaination: This table will have detailed information which is not covered in 3 main reports of financial statment. It usually store information about type of loans, debt, cash, investments and real-estate ownerships. 
-- Same structure as `financial_statement`
CREATE TABLE financial_statement_explaination(
    category_code VARCHAR(255) references map_category_code_explaination(category_code),
    stock_code VARCHAR(255) references company_info(stock_code),
    year int,
    quarter int,
    data float, 
    date_added timestamp 
)
