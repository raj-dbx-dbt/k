# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC ### Snowflake → Databricks full-catalog migration
# MAGIC
# MAGIC Reads a CSV with `source_catalog,destination_catalog` pairs and copies
# MAGIC everything under each source catalog into the matching destination,
# MAGIC creating the destination catalog/schemas as needed.
# MAGIC
# MAGIC - **Tables** always overwrite: `CREATE OR REPLACE TABLE`
# MAGIC - **Views** are recreated with their DDL rewritten to point at the
# MAGIC   destination catalog, so the destination becomes self-contained
# MAGIC - **System schemas** like `information_schema` are skipped

# COMMAND ----------

MAPPING_CSV  = "/Volumes/main/dbx_inventory/migration/catalogs.csv"
PARALLELISM  = 4
LOG_TABLE    = "dbx_inventory.migration_runs"
SKIP_SCHEMAS = {"information_schema", "pg_catalog", "sys"}

# COMMAND ----------

import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType,
)

run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
print(f"snowflake → databricks migration  |  {run_ts}")

# COMMAND ----------

# MAGIC %md ### 1 — read catalog pairs

# COMMAND ----------

m = pd.read_csv(MAPPING_CSV, dtype=str).fillna("")
m.columns = [c.strip().lower() for c in m.columns]

required = {"source_catalog", "destination_catalog"}
if required - set(m.columns):
    raise ValueError(f"csv missing columns: {required - set(m.columns)}")

m = m[(m["source_catalog"].str.strip() != "") &
      (m["destination_catalog"].str.strip() != "")]

pairs = m[["source_catalog", "destination_catalog"]].to_dict("records")

print(f"  {len(pairs)} catalog pair(s):")
for p in pairs:
    print(f"    {p['source_catalog']}  →  {p['destination_catalog']}")

# COMMAND ----------

def fq(*parts):
    return ".".join(f"`{p}`" for p in parts)

def plain(*parts):
    return ".".join(parts)

def rows_to_dicts(query):
    return [r.asDict() for r in spark.sql(query).collect()]

def is_view(src_fq):
    """Type row in DESCRIBE TABLE EXTENDED tells us VIEW vs MANAGED/EXTERNAL."""
    try:
        for r in rows_to_dicts(f"DESCRIBE TABLE EXTENDED {src_fq}"):
            if (r.get("col_name") or "").strip().lower() == "type":
                return (r.get("data_type") or "").strip().upper() == "VIEW"
        return False
    except Exception:
        return False

def rewrite_view_ddl(ddl, src_cat, dst_cat):
    """Repoint a view's DDL from the source catalog to the destination so
    the recreated view is self-contained on the UC side."""
    # backtick-quoted: `src_cat`  →  `dst_cat`
    ddl = ddl.replace(f"`{src_cat}`", f"`{dst_cat}`")
    # unquoted prefix: src_cat.  →  dst_cat.   (avoid mid-word collisions)
    ddl = re.sub(rf"(?<![\w`]){re.escape(src_cat)}\.", f"{dst_cat}.", ddl)
    # idempotent rerun-safety
    ddl = re.sub(r"^\s*CREATE\s+VIEW", "CREATE OR REPLACE VIEW",
                 ddl, count=1, flags=re.IGNORECASE)
    return ddl

# COMMAND ----------

# MAGIC %md ### 2 — discover schemas + objects

# COMMAND ----------

# walk the source side: SHOW SCHEMAS / SHOW TABLES, skipping built-ins.
# ensure each destination catalog + schema exists as we go.

objects = []
for p in pairs:
    src_cat, dst_cat = p["source_catalog"], p["destination_catalog"]

    try:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS `{dst_cat}`")
        print(f"  catalog ready: {dst_cat}")
    except Exception as ex:
        print(f"  !! catalog {dst_cat}: {ex}")
        continue

    try:
        schemas = [
            d["databaseName"]
            for d in rows_to_dicts(f"SHOW SCHEMAS IN `{src_cat}`")
            if d["databaseName"].lower() not in SKIP_SCHEMAS
        ]
    except Exception as ex:
        print(f"  !! could not list schemas in {src_cat}: {ex}")
        continue

    print(f"  {src_cat}: {len(schemas)} schemas")

    for sch in schemas:
        try:
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{dst_cat}`.`{sch}`")
        except Exception as ex:
            print(f"    !! schema {dst_cat}.{sch}: {ex}")
            continue

        try:
            for r in rows_to_dicts(f"SHOW TABLES IN `{src_cat}`.`{sch}`"):
                objects.append({
                    "src_cat": src_cat, "dst_cat": dst_cat,
                    "schema":  sch,     "name":    r["tableName"],
                })
        except Exception as ex:
            print(f"    !! could not list tables in {src_cat}.{sch}: {ex}")

print(f"\n  {len(objects)} objects to copy")

# COMMAND ----------

# MAGIC %md ### 3 — copy objects (tables + views)

# COMMAND ----------

# tables  →  CREATE OR REPLACE TABLE dst AS SELECT * FROM src   (full snapshot)
# views   →  pull source DDL, rewrite src_cat→dst_cat, run CREATE OR REPLACE VIEW
#
# rewriting view references makes the destination self-contained — once the
# tables it depends on have been copied, the view no longer needs the
# federated source to resolve.

def copy_object(o):
    src_fq    = fq(o["src_cat"], o["schema"], o["name"])
    dst_fq    = fq(o["dst_cat"], o["schema"], o["name"])
    dst_plain = plain(o["dst_cat"], o["schema"], o["name"])

    try:
        if is_view(src_fq):
            ddl_rows = rows_to_dicts(f"SHOW CREATE TABLE {src_fq}")
            if not ddl_rows:
                raise RuntimeError("SHOW CREATE TABLE returned nothing")
            ddl_text = (ddl_rows[0].get("createtab_stmt")
                        or ddl_rows[0].get("CREATE_TABLE_STMT")
                        or "")
            if not ddl_text:
                raise RuntimeError("could not extract view DDL")

            new_ddl = rewrite_view_ddl(ddl_text, o["src_cat"], o["dst_cat"])
            spark.sql(new_ddl)
            print(f"  view: {dst_plain}")
            return {"table": dst_plain, "status": "view_created",
                    "rows": None, "error": None}

        spark.sql(f"CREATE OR REPLACE TABLE {dst_fq} AS SELECT * FROM {src_fq}")
        rows = spark.sql(f"SELECT COUNT(*) AS n FROM {dst_fq}").collect()[0]["n"]
        print(f"  table: {dst_plain}  ({rows:,} rows)")
        return {"table": dst_plain, "status": "table_created",
                "rows": rows, "error": None}

    except Exception as ex:
        print(f"  !! {dst_plain}: {ex}")
        return {"table": dst_plain, "status": "failed",
                "rows": None, "error": str(ex)}


results = []
with ThreadPoolExecutor(max_workers=PARALLELISM) as pool:
    futures = {pool.submit(copy_object, o): o for o in objects}
    for fut in as_completed(futures):
        results.append(fut.result())

# COMMAND ----------

# MAGIC %md ### 4 — summary + log

# COMMAND ----------

tally = lambda s: sum(1 for r in results if r["status"] == s)

print(f"\n{'─'*60}")
print(f"  tables  : {tally('table_created')}")
print(f"  views   : {tally('view_created')}")
print(f"  failed  : {tally('failed')}")
print(f"{'─'*60}")

for r in (r for r in results if r["status"] == "failed"):
    print(f"\n  failed: {r['table']}\n    {r['error']}")

LOG_SCHEMA = StructType([
    StructField("table",       StringType(), True),
    StructField("status",      StringType(), True),
    StructField("rows",        LongType(),   True),
    StructField("error",       StringType(), True),
    StructField("migrated_at", StringType(), True),
])

(spark.createDataFrame(
    [(r["table"], r["status"], r["rows"], r["error"], run_ts) for r in results],
    LOG_SCHEMA,
)
 .write.format("delta").mode("append").option("mergeSchema", "true")
 .saveAsTable(LOG_TABLE))

print(f"\nlog → {LOG_TABLE}")
print(f"done  |  {run_ts}")
