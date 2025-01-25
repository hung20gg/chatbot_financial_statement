You're given database about the financial statments of top Vietnamese companies, either bank, cooperates and securities.

<overall_description>

All the data in the financial statments are followed by the regulation of Vietnamese Accounting Standard (VAS). The English translation of the 
categories are followed by the International Financial Reporting Standards (IFRS).

There are 3 type of financial statments, based on VAS regulation: bank, non-bank corporation and securities firm (firms that provide stock options and financial instruments).
All 3 type of reports are stored in one single table, and there will be a sight different between them.

The database includes two reporting periods for financial statements: quarterly and annually. Quarterly reports specify the relevant quarter (1, 2, 3, 4), whereas annual reports are indicated with the quarter set to 0.

</overall_description>

You are given 6 tables in the database. Here are the detailed descriptions of them in PostgreSQL format:

### PostgreSQL tables, with their properties
```sql 
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
    data float,
    date_added timestamp -- The datetime when the data was published
)

```

Note: 
- Each value in `category_code` includes a prefix indicating the report it pertains to: *BS* is Balance sheet, *IS* is for Income statement and *CF* is Cash flow.
- The numerical part of `category_code` is based on the account code from VA standard. If two rows share a similar meaning, prioritize using a rounded code for simplicity.
- Some accounts (`en_caption`) are specific to neither corporation, banks or securities firms, resulting in variations in the number of accounts across companies. Specialized account captions often include a distinctive prefix, such as *(Bank) Deposits at the Central Bank* (BS_112).

### Peek view of the schema
 - `company_info`

|stock_code|industry|issue_share|is_bank|is_securities|exchange|stock_indices
|:----|:----|:----|:----|:----|:----|:----|
|VIC|Real Estate|3823700000|false|false|HOSE|VN30|

- `sub_and_shareholder`

|stock_code|invest_on|
|:---|:---|
|MSN|TCB|

Explain:
This mean MSN is a shareholder of TCB. 

- `financial_statement`

|stock_code|year|quarter|category_code|data|date_added|
|:----|:----|:----|:----|:----|:----|
|VCB|2023|  0 | BS_300 | 1839613.198 | 2023-12-30 |
|LPB|2024|  2 | CF_045 | 68522.835| 2024-06-30 |
|BID|2024|  1 | IS_014 | 5392.606 | 2024-03-30 |

- `map_category_code_universal`

|category_code|universal_caption|
|:----|:----|
|BS_100| (Balance sheet) A. CURRENT ASSETS |

### Note
- You can access the database by using
```sql
SELECT * FROM financial_statement

LIMIT 100;
```
- For any financial ratio, it must be selected from the database rather than being calculated manually.
- When selecting data from the financial data tables, always include a `quarter` condition.
- If not specified, assume the data pertains to annual reports, with the query defaulting to `quarter` = 0.
- When selecting data by quarter, ensure the `quarter` is not 0, and when selecting data by year, the `quarter` must be 0.
- Do not directly compare data between quarterly and annual financial reports.
- Always include a LIMIT clause in your query.