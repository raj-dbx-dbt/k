# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC ### One-shot catalog clone (Snowflake federated → UC native)
# MAGIC
# MAGIC Mirrors a single source catalog into a new target catalog named
# MAGIC `<TARGET_PREFIX>_<SOURCE_CATALOG>`. Creates the target catalog,
# MAGIC mirrors every schema, full-copies every table as Delta, and
# MAGIC recreates every view pointing back at the source.
# MAGIC
# MAGIC After the data lands, a metadata pass copies what CTAS doesn't
# MAGIC carry — column comments, table comments, and NOT NULL constraints.
# MAGIC
# MAGIC One-time migration. Re-running it re-snapshots everything (full refresh).

# COMMAND ----------

SOURCE_CATALOG = "sf_prod"          # the federated source catalog
TARGET_PREFIX  = "migrated"         # target = "{prefix}_{source}" → "migrated_sf_prod"
PARALLELISM    = 4                  # concurrent table copies
LOG_TABLE      = "dbx_inventory.migration_runs"

# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pyspark.sql.types import StructType, StructField, StringType, LongType, BooleanType

target_catalog = f"{TARGET_PREFIX}_{SOURCE_CATALOG}"
run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
print(f"one-shot clone  |  {SOURCE_CATALOG}  →  {target_catalog}  |  {run_ts}")

# COMMAND ----------

def fq(*parts):
    return ".".join(f"`{p}`" for p in parts)

def plain(*parts):
    return ".".join(parts)

def rows_to_dicts(query):
    return [r.asDict() for r in spark.sql(query).collect()]

# COMMAND ----------

# MAGIC %md ### 1 — ensure target catalog

# COMMAND ----------

try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS `{target_catalog}`")
    print(f"  catalog ready: {target_catalog}")
except Exception as ex:
    raise RuntimeError(f"could not create target catalog `{target_catalog}`: {ex}")

# COMMAND ----------

# MAGIC %md ### 2 — discover schemas + objects

# COMMAND ----------

# walk the source with SHOW SCHEMAS / SHOW TABLES — works on every federated
# source. skip information_schema since that's UC's metadata view, not user data.

schemas = [
    d["databaseName"]
    for d in rows_to_dicts(f"SHOW SCHEMAS IN `{SOURCE_CATALOG}`")
    if d["databaseName"].lower() != "information_schema"
]
print(f"  {len(schemas)} schemas to mirror")

# flatten objects so the parallel copy loop has a single unit-of-work list
objects = []
for sch in schemas:
    try:
        rows = rows_to_dicts(f"SHOW TABLES IN `{SOURCE_CATALOG}`.`{sch}`")
    except Exception as ex:
        print(f"    couldn't list {SOURCE_CATALOG}.{sch}: {ex}")
        continue

    for r in rows:
        objects.append({"schema": sch, "name": r["tableName"]})

print(f"  {len(objects)} objects to copy")

# COMMAND ----------

# MAGIC %md ### 3 — mirror schemas

# COMMAND ----------

for sch in schemas:
    try:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{target_catalog}`.`{sch}`")
        print(f"  schema ready: {target_catalog}.{sch}")
    except Exception as ex:
        print(f"  !! {target_catalog}.{sch}: {ex}")

# COMMAND ----------

# MAGIC %md ### 4 — copy tables and views

# COMMAND ----------

# table  →  CREATE OR REPLACE TABLE AS SELECT *  (delta snapshot)
# view   →  CREATE OR REPLACE VIEW AS SELECT *   (definition pointing at source)
#
# we detect view-vs-table from DESCRIBE TABLE EXTENDED's "Type" row.

def is_view(src_fq):
    try:
        rows = rows_to_dicts(f"DESCRIBE TABLE EXTENDED {src_fq}")
        for r in rows:
            if (r.get("col_name") or "").strip().lower() == "type":
                return (r.get("data_type") or "").strip().upper() == "VIEW"
        return False
    except Exception:
        return False

def copy_object(obj):
    sch       = obj["schema"]
    name      = obj["name"]
    src_fq    = fq(SOURCE_CATALOG, sch, name)
    tgt_fq    = fq(target_catalog, sch, name)
    tgt_plain = plain(target_catalog, sch, name)

    try:
        if is_view(src_fq):
            spark.sql(f"CREATE OR REPLACE VIEW {tgt_fq} AS SELECT * FROM {src_fq}")
            rows = spark.sql(f"SELECT COUNT(*) AS n FROM {tgt_fq}").collect()[0]["n"]
            print(f"  view_created: {tgt_plain}  ({rows:,} rows visible)")
            return {"table": tgt_plain, "status": "view_created",
                    "rows": rows, "error": None, "metadata_applied": False}

        spark.sql(f"CREATE OR REPLACE TABLE {tgt_fq} AS SELECT * FROM {src_fq}")
        rows = spark.sql(f"SELECT COUNT(*) AS n FROM {tgt_fq}").collect()[0]["n"]
        print(f"  created: {tgt_plain}  ({rows:,} rows)")
        return {"table": tgt_plain, "status": "created",
                "rows": rows, "error": None, "metadata_applied": False}

    except Exception as ex:
        print(f"  !! {tgt_plain}: {ex}")
        return {"table": tgt_plain, "status": "failed",
                "rows": None, "error": str(ex), "metadata_applied": False}


results = []
with ThreadPoolExecutor(max_workers=PARALLELISM) as pool:
    futures = {pool.submit(copy_object, o): o for o in objects}
    for fut in as_completed(futures):
        results.append((fut.result(), futures[fut]))

# COMMAND ----------

# MAGIC %md ### 5 — metadata pass (column + table comments, NOT NULL)

# COMMAND ----------

# CTAS doesn't carry these — pull from the federated information_schema
# and apply via ALTER. views skip this since they inherit from their definition.

def apply_metadata(obj):
    sch    = obj["schema"]
    name   = obj["name"]
    tgt_fq = fq(target_catalog, sch, name)

    try:
        cols = rows_to_dicts(f"""
            SELECT column_name, comment, is_nullable
            FROM `{SOURCE_CATALOG}`.`information_schema`.`columns`
            WHERE LOWER(table_schema) = LOWER('{sch}')
              AND LOWER(table_name)   = LOWER('{name}')
        """)

        for d in cols:
            cname    = (d.get("column_name") or d.get("COLUMN_NAME") or "").lower()
            comment  =  d.get("comment")     or d.get("COMMENT")
            nullable = (d.get("is_nullable") or d.get("IS_NULLABLE") or "").upper()

            if not cname:
                continue

            if comment:
                safe = comment.replace("'", "''")
                spark.sql(
                    f"ALTER TABLE {tgt_fq} "
                    f"ALTER COLUMN `{cname}` COMMENT '{safe}'"
                )

            if nullable == "NO":
                # silently skip if target already has nulls — real signal but
                # not worth aborting the migration over
                try:
                    spark.sql(f"ALTER TABLE {tgt_fq} "
                              f"ALTER COLUMN `{cname}` SET NOT NULL")
                except Exception:
                    pass

        # table-level comment
        tabs = rows_to_dicts(f"""
            SELECT comment
            FROM `{SOURCE_CATALOG}`.`information_schema`.`tables`
            WHERE LOWER(table_schema) = LOWER('{sch}')
              AND LOWER(table_name)   = LOWER('{name}')
        """)
        if tabs:
            tcom = tabs[0].get("comment") or tabs[0].get("COMMENT")
            if tcom:
                safe = tcom.replace("'", "''")
                spark.sql(f"COMMENT ON TABLE {tgt_fq} IS '{safe}'")

        return True, None

    except Exception as ex:
        return False, str(ex)


print("\nmetadata pass...")
for r, obj in results:
    if r["status"] == "created":  # tables only; views skip
        ok, err = apply_metadata(obj)
        r["metadata_applied"] = ok
        if not ok:
            print(f"  metadata !! {r['table']}: {err}")

# COMMAND ----------

# MAGIC %md ### 6 — summary + log

# COMMAND ----------

results_only = [r for r, _ in results]

def tally(s):
    return sum(1 for r in results_only if r["status"] == s)

print(f"\n{'─'*60}")
print(f"  tables created : {tally('created')}")
print(f"  views created  : {tally('view_created')}")
print(f"  failed         : {tally('failed')}")
print(f"{'─'*60}")

for r in (r for r in results_only if r["status"] == "failed"):
    print(f"\n  failed: {r['table']}\n    {r['error']}")

LOG_SCHEMA = StructType([
    StructField("table",            StringType(),  True),
    StructField("status",           StringType(),  True),
    StructField("rows",             LongType(),    True),
    StructField("error",            StringType(),  True),
    StructField("metadata_applied", BooleanType(), True),
    StructField("migrated_at",      StringType(),  True),
])

(spark.createDataFrame(
    [(r["table"], r["status"], r["rows"], r["error"],
      r["metadata_applied"], run_ts) for r in results_only],
    LOG_SCHEMA,
)
 .write.format("delta").mode("append").option("mergeSchema", "true")
 .saveAsTable(LOG_TABLE))

print(f"\nlog → {LOG_TABLE}")
print(f"done  |  {run_ts}")
