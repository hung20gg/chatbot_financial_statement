You're given database about the financial statments of top Vietnamese companies, either bank, cooperates and securities.

<overall_description>

All the data in the financial statments are followed by the regulation of Vietnamese Accounting Standard (VAS). The English translation of the 
categories are followed by the International Financial Reporting Standards (IFRS).

There are 3 type of financial statments, based on VAS regulation: bank, non-bank corporation and securities firm (firms that provide stock options and financial instruments).
All 3 type of reports are stored in one single table, and there will be a sight different between them.

The database includes two reporting periods for financial statements: quarterly and annually. Quarterly reports specify the relevant quarter (1, 2, 3, 4), whereas annual reports are indicated with the quarter set to 0.

</overall_description>

Here are the detailed descriptions of them in PostgreSQL format:

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


-- Table: universal_financial_report_hori: Universal financial report 
CREATE TABLE universal_financial_report_hori(
    stock_code VARCHAR(255) references company_info(stock_code),
    year int, -- The reported financial year
    quarter int, --  The quarter reported (contain value either 0, 1, 2, 3, 4). If the value is 0, that mean the report is for annual report.
    "BS_100" float, -- The column name corresponds to the code in balance sheet universal standard. The unit of data in this column is always in Million VND.
    -- ... 
    "CF_001" float, -- The column name corresponds to the code in cashflow statement universal standard. The unit of data in this column is always in Million VND.
    --- ...
    "IS_001" float -- The column name corresponds to the code in income statement universal standard. The unit of data in this column is always in Million VND.
    -- ...
);


-- Table financial_ratio: This table will have pre-calculated common Financial Ratio such as ROA, ROE, FCF, etc
-- Same structure as `financial_statement`
CREATE TABLE financial_ratios_hori(
    stock_code VARCHAR(255) references company_info(stock_code),
    year int,
    quarter int,
    "ROE" float -- financial ratio
    -- ...
)

```

Note: 
- For each column name in universal_financial_report_hori , the prefix indicating the report it pertains to: *BS* is Balance sheet, *IS* is for Income statement and *CF* is Cash flow.
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

- `universal_financial_report_hori`

|stock_code|year|quarter|BS_100|...|IS_001|...|CF_001|
|:---|:----|:----|:----|:----|:----|:----|:----|
|VCB|2023|  0 | 110000|...|7837|...| 1839613.198 |

### Note
- You can access the database by using
```sql
SELECT * FROM universal_financial_report_hori

LIMIT 100;
```
- Column name like "BS_110" needs to be in "" when query.
- The data in columns like "BS_110" from the tables bank_financial_report_hori, non_bank_financial_report_hori, and sec_financial_report_hori is always recorded in millions of VND. Therefore, ensure that you take this unit into account and convert it properly before performing any calculations.
- For any financial ratio, it must be selected from the database rather than being calculated manually.
- When selecting data from the financial data tables, always include a `quarter` condition.
- If not specified, assume the data pertains to annual reports, with the query defaulting to `quarter` = 0.
- When selecting data by quarter, ensure the `quarter` is not 0, and when selecting data by year, the `quarter` must be 0.
- Do not directly compare data between quarterly and annual financial reports.
- Always include a LIMIT clause in your query.