You're given database about the financial statments of top Vietnamese firms.

<overall_description>

All the data in the financial statments are followed by the regulation of Vietnamese Accounting Standard (VAS).

There are 3 type of financial statments, based on VAS regulation: bank, corporation and securities firm. All 3 type of reports are stored in one single table, and there will be a sight different between them.

The database includes two reporting periods: quarterly and annually. Quarterly reports specify the relevant quarter (1, 2, 3, 4), whereas annual reports are indicated with the quarter set to 0.

</overall_description>

### PostgreSQL tables, with their properties
```sql 
-- Table: company_info
CREATE TABLE company_info(
    stock_code VARCHAR(255) primary key, --The trading symbol.
    industry VARCHAR(255), 
    exchange VARCHAR(255), -- The listed exchange of stock
    stock_indices VARCHAR(255), -- The stock index it belongs to (e.g., VN30, HNX30)
    is_bank BOOLEAN,
    is_securities BOOLEAN,
);

--  /* Snapshot 

--  stock_code industry  is_bank  is_securities  exchange  stock_indices 
--   VIC  Real Estate  false  false  HOSE  VN30  
-- */

-- Table: sub_and_shareholder
CREATE TABLE sub_and_shareholder(
    stock_code VARCHAR(255) NOT NULL, 
    invest_on VARCHAR(255) NOT NULL, -- The company invested on (can be subsidiary)
    FOREIGN KEY (stock_code) REFERENCES company_info(stock_code),
    FOREIGN KEY (invest_on) REFERENCES company_info(stock_code),
    PRIMARY KEY (stock_code, invest_on) 
);

-- /*  MSN is a shareholder of TCB.

-- stock_code  invest_on
-- MSN  TCB
-- */

-- Table: map_category_code_universal: Mapping account table for Financial Statement
CREATE TABLE map_category_code_universal(
    category_code VARCHAR(255) primary key, --The unique code for accounts recorded.
    en_caption VARCHAR(255), --The account's captions.
);



-- Table: financial_statement: Financial Statement data for each `stock_code`
CREATE TABLE financial_statement(
    stock_code VARCHAR(255) references company_info(stock_code),
    year int, -- The reported financial year
    quarter int, --  The quarter reported (either 0, 1, 2, 3, 4). 
    category_code VARCHAR(255) references map_category_code_universal(category_code),
    data float, -- in Million VND except ratio
    date_added timestamp
);

-- /*
-- stock_code  year  quarter  category_code  data  date_added 
--  VCB  2023   0  BS_300  1839613.198  2023-12-30 
--  LPB  2024   2  CF_045  68522.835  2024-06-30 
-- */

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


-- Table map_category_code_ratio
CREATE TABLE map_category_code_ratio(
    ratio_code VARCHAR(255) primary key,
    ratio_name VARCHAR(255)
);

-- /*
-- category_code  universal_caption
-- BS_100  '(Balance sheet) A. CURRENT ASSETS'
-- */

-- Table financial_ratio: This table will have pre-calculated common Financial Ratio such as ROA, ROE, FCF, etc
-- Same structure as `financial_statement`
CREATE TABLE financial_ratio(
    ratio_code VARCHAR(255) references map_category_code_ratio(ratio_code),
    stock_code VARCHAR(255) references company_info(stock_code),
    year int,
    quarter int,
    data float, -- Either in Million VND if money, or ratio otherwise
    date_added timestamp
)

-- Table: industry_financial_ratio
CREATE TABLE industry_financial_ratio(
    industry VARCHAR(255),
    ratio_code VARCHAR(255) references map_category_code_ratio(ratio_code),
    year int,
    quarter int,
    data_mean float, 
    date_added timestamp
)

-- Table map_category_code_explaination
CREATE TABLE map_category_code_explaination(
    category_code VARCHAR(255) primary key,
    en_caption VARCHAR(255),
);

-- Table financial_statement_explaination: Contain information which is not covered in 3 main reports. It usually stores information about type of loans, debt, cash, investments and real-estate ownerships. 

-- Same structure as `financial_statement`
CREATE TABLE financial_statement_explaination(
    category_code VARCHAR(255) references map_category_code_explaination(category_code),
    stock_code VARCHAR(255) references company_info(stock_code),
    year int,
    quarter int,
    data float, 
    date_added timestamp 
)
```

### Note on schema description: 
- Each value in `category_code` includes a prefix indicating the report it pertains to: *BS* is Balance sheet, *IS* is for Income statement, *CF* is Cash flow and *TM* is for Explaination.
- For `category_code` in `map_category_code_explaination`, there are 4 additional prefix: *Crop*, *Bank*, *Sec* and *Share* for specific type of organization.
- If two rows share a similar meaning, using a rounded code.
- With YoY ratio in `financial_ratio`, you should recalculate the ratio if the time window is not 1 year.

### Note on creating query:
- For any financial ratio, it must be selected from the database rather than being calculated manually.
- Always include a `quarter` condition in your query. If not specified, assume using annual reports, with the query defaulting to `quarter` = 0.
- Always include a LIMIT clause in your query.