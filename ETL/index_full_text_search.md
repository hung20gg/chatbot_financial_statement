After pushing data to db, add the bm25 index for `industry` in `company_info` table

```sql

-- Drop dependent objects first
DROP TRIGGER IF EXISTS industry_tsvector_trigger ON company_info;
DROP FUNCTION IF EXISTS update_industry_tsvector();
DROP INDEX IF EXISTS industry_tsvector_idx;
ALTER TABLE company_info DROP COLUMN IF EXISTS industry_tsvector;

-- Add column
ALTER TABLE company_info
ADD COLUMN industry_tsvector tsvector;

-- Update existing data
UPDATE company_info
SET industry_tsvector = to_tsvector('english', industry);

-- Create index
CREATE INDEX industry_tsvector_idx
ON company_info
USING GIN (industry_tsvector);

-- Create function
CREATE FUNCTION update_industry_tsvector() RETURNS trigger AS $$
BEGIN
  NEW.industry_tsvector := to_tsvector('english', NEW.industry);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
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