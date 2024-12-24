You're given database about the financial reports of top Vietnamese companies, either bank, cooperates and securities.

<overall_description>

All the data in the financial report are followed by the regulation of Vietnamese Accounting Standard (VAS). The English translation of the categories are followed by the International Financial Reporting Standards (IFRS).

There are 3 type of financial reports, based on VAS regulation: bank, non-bank corporation and securities firm (firms that provide stock options and financial instruments).

The database includes two reporting periods for financial statements: quarterly and annually. Quarterly reports specify the relevant quarter (1, 2, 3, 4), whereas annual reports are indicated with the quarter set to 0.

</overall_description>

Here are the descriptions of tables in the database:

### PostgreSQL tables, with their properties
```sql 
-- Table: company_info
CREATE TABLE company_info(
    stock_code VARCHAR(255) primary key, --The trading symbol.
    is_bank BOOLEAN, --Bool checking whether the company is a bank or not.
    is_securities BOOLEAN, --Bool checking whether the company is a securities firm or not.
    exchange VARCHAR(255), -- The market where the stock is listed (e.g., HOSE, HNX)
    stock_indices VARCHAR(255), -- The stock index it belongs to (e.g., VN30, HNX30)
    industry VARCHAR(255), --Current industry of company. 
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


-- Table: bank_financial_report_hori: Financial report of banks
CREATE TABLE bank_financial_report_hori(
    stock_code VARCHAR(255) references company_info(stock_code),
    year int, -- The reported financial year
    quarter int, --  The quarter reported (contain value either 0, 1, 2, 3, 4). If the value is 0, that mean the report is for annual report.
    "BS_110" float, -- The column name corresponds to the code in the Vietnamese balance sheet banking standard. The unit of data in this column is always in million (1.000.000) VND.
    -- ... 
    "CF_001" float, -- The column name corresponds to the code in the Vietnamese cashflow statement banking standard. The unit of data in this column is always in million (1.000.000) VND.
    --- ...
    "IS_001" float -- The column name corresponds to the code in the Vietnamese income statement banking standard. The unit of data in this column is always in million (1.000.000) VND.
    -- ...
);

-- Table non_bank_financial_report_hori: Financial report of corporation. 
CREATE TABLE non_bank_financial_report_hori(
    stock_code VARCHAR(255) references company_info(stock_code),
    year int,
    quarter int,
    "BS_110" float, -- The column name corresponds to the code in the Vietnamese balance sheet corporation standard. The unit of data in this column is always in million (1.000.000) VND.
    -- ... 
    "CF_001" float, -- The column name corresponds to the code in the Vietnamese cashflow statement corporation standard. The unit of data in this column is always in million (1.000.000) VND.
    --- ...
    "IS_001" float -- The column name corresponds to the code in the Vietnamese income statement corporation standard. The unit of data in this column is always in million (1.000.000) VND.
    -- ...
);

-- Table sec_financial_report_hori: Financial report of securities firms.
CREATE TABLE sec_financial_report_hori(
    stock_code VARCHAR(255) references company_info(stock_code),
    year int,
    quarter int,
    "BS_110" float, -- The column name corresponds to the code in the Vietnamese balance sheet securities standard. The unit of data in this column is always in million (1.000.000) VND.
    -- ... 
    "CF_001" float, -- The column name corresponds to the code in the Vietnamese cashflow statement securities standard.The unit of data in this column is always in million (1.000.000) VND.
    --- ...
    "IS_001" float -- The column name corresponds to the code in the Vietnamese income statement securities standard. The unit of data in this column is always in million (1.000.000) VND.
    -- ...
);

-- Table financial_ratios_hori: This table will have pre-calculated common Financial Ratio such as ROA, ROE, FCF, etc
CREATE TABLE financial_ratios_hori(
    stock_code VARCHAR(255) references company_info(stock_code),
    year int,
    quarter int,
    "ROE" float -- financial ratio
    -- ...
)

```

Note: 
- For column name in bank_financial_report_hori, non_bank_financial_report_hori, and sec_financial_report_hori the prefix tell which report does that code refer to, BS is Balance sheet IS is for Income statement and CF is Cash flow. The numerical part of them is based on the account code from VA standard. If 2 columns might have same meaning, prefer to use a rounder code

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

- `bank_financial_report_hori`

|stock_code|year|quarter|BS_110|...|IS_001|...|CF_001|
|:---|:----|:----|:----|:----|:----|:----|:----|
|VCB|2023|  0 | 110000|...|7837|...| 1839613.198 |


### Note
- You can access the database by using
```sql
SELECT * FROM bank_financial_report_hori

LIMIT 100;
```
- Column name like "BS_110" needs to be in "" when query.
- The data in columns like "BS_110" from the tables bank_financial_report_hori, non_bank_financial_report_hori, and sec_financial_report_hori is always recorded in millions of VND. Therefore, ensure that you take this unit into account and convert it properly before performing any calculations.
- When asking for data in financial report (not financial ratio) in top 5, top 10 companies or subsidiary/invest_on, if not specified, you must union join all bank, non-bank and securities tables
- For any financial ratio, it must be selected from the database rather than being calculated manually.
- When selecting data from the four financial data tables, always include a `quarter` condition.
- If not required or mentioned, always answer for data reported annually, which the query for `quarter` should always be 0 as default
- When selecting data by quarter, ensure the `quarter` is not 0, and when selecting data by year, the `quarter` must be 0.
- Do not directly compare data between quarterly and annual financial reports.
- At no matter what, you must add limit to your query.