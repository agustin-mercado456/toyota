CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'mlflow';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;

CREATE DATABASE toyota_db;
DO $$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles WHERE rolname = 'user'
   ) THEN
      CREATE USER user WITH ENCRYPTED PASSWORD 'airbyte';
   END IF;
END
$$;
GRANT ALL PRIVILEGES ON DATABASE mlops TO "user";
GRANT ALL ON SCHEMA public TO "user";
GRANT USAGE ON SCHEMA public TO "user";
ALTER DATABASE mlops OWNER TO "user";

