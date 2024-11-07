After pushing data to db, add the bm25 index for `industry` in `company_info` table

```sql

ALTER TABLE company_info
ADD COLUMN industry_tsvector tsvector;

UPDATE company_info
SET industry_tsvector = to_tsvector('english', industry);

CREATE INDEX industry_tsvector_idx
ON company_info
USING GIN (industry_tsvector);


CREATE FUNCTION update_industry_tsvector() RETURNS trigger AS $$
BEGIN
  NEW.industry_tsvector := to_tsvector('english', NEW.industry);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER industry_tsvector_trigger
BEFORE INSERT OR UPDATE ON company_info
FOR EACH ROW EXECUTE FUNCTION update_industry_tsvector();

```

After that, you can query for `industry` as follow

```sql
SELECT industry
FROM company_info
WHERE industry_tsvector @@ to_tsquery('english', 'technology');
```