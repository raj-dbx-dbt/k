# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC ### Snowflake → Databricks all-up migration
# MAGIC
# MAGIC One notebook that:
# MAGIC 1. Creates the inventory schema, the mapping volume, the log table
# MAGIC 2. Writes a starter mapping CSV if you don't have one
# MAGIC 3. Tests the snowflake connection
# MAGIC 4. Discovers schemas + objects in every source database
# MAGIC 5. Pre-flight scans view dependencies and warns about externals
# MAGIC 6. Copies every table (overwrite) and every view (recreated)
# MAGIC 7. Logs the run
# MAGIC
# MAGIC **First run:** writes a starter CSV and exits — edit it, then re-run.
# MAGIC **Subsequent runs:** does the migration end-to-end.
# MAGIC
# MAGIC References:
# MAGIC - Snowflake Spark connector — https://docs.snowflake.com/en/user-guide/spark-connector-use
# MAGIC - Databricks/Snowflake integration — https://docs.databricks.com/aws/en/connect/external-systems/snowflake

# COMMAND ----------

# MAGIC %md ### Config

# COMMAND ----------

# ── snowflake connection ─────────────────────────────────────────────────────

# find the account URL in snowsight under Admin → Accounts.
# format: <account_identifier>.snowflakecomputing.com
SF_URL       = "xy12345.us-east-1.snowflakecomputing.com"
SF_WAREHOUSE = "COMPUTE_WH"
SF_ROLE      = "ACCOUNTADMIN"

# credentials live in databricks secrets. one-time setup from your terminal:
#   databricks secrets create-scope snowflake
#   databricks secrets put-secret snowflake username
#   databricks secrets put-secret snowflake password
SF_SECRET_SCOPE = "snowflake"

# ── databricks workspace ─────────────────────────────────────────────────────

INVENTORY_CATALOG = "main"
INVENTORY_SCHEMA  = "dbx_inventory"
INVENTORY_VOLUME  = "migration"
MAPPING_FILENAME  = "databases.csv"

# ── migration behavior ───────────────────────────────────────────────────────

PARALLELISM  = 4                          # concurrent copies
SKIP_SCHEMAS = {"INFORMATION_SCHEMA"}     # snowflake system schemas

# COMMAND ----------

import os
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pyspark.sql.types import StructType, StructField, StringType, LongType

LOG_TABLE   = f"{INVENTORY_CATALOG}.{INVENTORY_SCHEMA}.migration_runs"
VOLUME_PATH = f"/Volumes/{INVENTORY_CATALOG}/{INVENTORY_SCHEMA}/{INVENTORY_VOLUME}"
MAPPING_CSV = f"{VOLUME_PATH}/{MAPPING_FILENAME}"
RUN_TS      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

print(f"databricks workspace setup  |  {RUN_TS}\n")

# COMMAND ----------

# MAGIC %md ### 1 — set up inventory schema, volume, log table

# COMMAND ----------

# all idempotent — safe to re-run
spark.sql(f"CREATE CATALOG IF NOT EXISTS `{INVENTORY_CATALOG}`")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS `{INVENTORY_CATALOG}`.`{INVENTORY_SCHEMA}`")
spark.sql(f"CREATE VOLUME  IF NOT EXISTS `{INVENTORY_CATALOG}`.`{INVENTORY_SCHEMA}`.`{INVENTORY_VOLUME}`")

print(f"  inventory schema : {INVENTORY_CATALOG}.{INVENTORY_SCHEMA}")
print(f"  inventory volume : {VOLUME_PATH}")
print(f"  log table        : {LOG_TABLE}")

# COMMAND ----------

# MAGIC %md ### 2 — ensure a mapping CSV exists

# COMMAND ----------

# if the CSV isn't there yet, write a starter the user can edit, then exit.
# next run picks it up from this same path.

if not os.path.exists(MAPPING_CSV):
    template = pd.DataFrame([
        {"source_database": "SALES_PROD",   "destination_catalog": "migrated_sales_prod"},
        {"source_database": "FINANCE_PROD", "destination_catalog": "migrated_finance_prod"},
    ])
    template.to_csv(MAPPING_CSV, index=False)
    print(f"\n  ⚠  starter CSV written to {MAPPING_CSV}")
    print(f"     edit it with your real database list, then re-run this notebook.")
    dbutils.notebook.exit("starter CSV created — edit it and re-run")
else:
    print(f"  mapping csv      : {MAPPING_CSV} ✓")

# COMMAND ----------

# MAGIC %md ### Snowflake connection helpers

# COMMAND ----------

# canonical option names per snowflake's official docs
def sf_opts(database=None, schema=None):
    opts = {
        "sfURL":       SF_URL,
        "sfUser":      dbutils.secrets.get(SF_SECRET_SCOPE, "username"),
        "sfPassword":  dbutils.secrets.get(SF_SECRET_SCOPE, "password"),
        "sfWarehouse": SF_WAREHOUSE,
        "sfRole":      SF_ROLE,
    }
    if database: opts["sfDatabase"] = database
    if schema:   opts["sfSchema"]   = schema
    return opts

def sf_query(database, sql, schema=None):
    return (spark.read
        .format("snowflake")
        .options(**sf_opts(database, schema))
        .option("query", sql)
        .load())

def sf_read_table(database, schema, table):
    return (spark.read
        .format("snowflake")
        .options(**sf_opts(database, schema))
        .option("dbtable", f'"{schema}"."{table}"')
        .load())

def fq(*parts):
    return ".".join(f"`{p}`" for p in parts)

def plain(*parts):
    return ".".join(parts)

# COMMAND ----------

# MAGIC %md ### 3 — test snowflake connection

# COMMAND ----------

# verify credentials + networking before doing any real work. if this fails,
# the migration would fail too — exit early with a useful error.

print(f"  testing snowflake connection...")
try:
    probe = sf_query("SNOWFLAKE",
        "SELECT CURRENT_VERSION() AS v, CURRENT_USER() AS u, CURRENT_ROLE() AS r"
    ).collect()[0].asDict()
    print(f"  ok — snowflake {probe['V']}  |  user={probe['U']}  role={probe['R']}")
except Exception as ex:
    msg = str(ex)
    print(f"  !! snowflake connection failed:\n     {msg}")
    if "secret" in msg.lower() or "scope" in msg.lower():
        print(f"\n  did you store the credentials? run from your terminal:")
        print(f"    databricks secrets create-scope {SF_SECRET_SCOPE}")
        print(f"    databricks secrets put-secret {SF_SECRET_SCOPE} username")
        print(f"    databricks secrets put-secret {SF_SECRET_SCOPE} password")
    dbutils.notebook.exit("snowflake connection failed — see error above")

# COMMAND ----------

# MAGIC %md ### 4 — read the CSV

# COMMAND ----------

m = pd.read_csv(MAPPING_CSV, dtype=str).fillna("")
m.columns = [c.strip().lower() for c in m.columns]

required = {"source_database", "destination_catalog"}
if required - set(m.columns):
    raise ValueError(f"csv missing columns: {required - set(m.columns)}")

m = m[(m["source_database"].str.strip() != "") &
      (m["destination_catalog"].str.strip() != "")]
pairs = m[["source_database", "destination_catalog"]].to_dict("records")

print(f"\n  {len(pairs)} database(s) to migrate")
for p in pairs:
    print(f"    {p['source_database']}  →  {p['destination_catalog']}")

# COMMAND ----------

# MAGIC %md ### 5 — discover schemas + objects

# COMMAND ----------

objects = []
for p in pairs:
    src_db, dst_cat = p["source_database"], p["destination_catalog"]

    try:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS `{dst_cat}`")
        print(f"  catalog ready: {dst_cat}")
    except Exception as ex:
        print(f"  !! catalog {dst_cat}: {ex}")
        continue

    try:
        rows = sf_query(src_db, f"""
            SELECT table_schema, table_name, table_type
            FROM {src_db}.information_schema.tables
            WHERE table_schema NOT IN ('{"','".join(SKIP_SCHEMAS)}')
            ORDER BY table_schema, table_name
        """).collect()
    except Exception as ex:
        print(f"  !! could not enumerate {src_db}: {ex}")
        continue

    schemas_seen = set()
    for r in rows:
        d    = r.asDict()
        sch  = d["TABLE_SCHEMA"]
        name = d["TABLE_NAME"]
        kind = d["TABLE_TYPE"]

        if sch not in schemas_seen:
            try:
                spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{dst_cat}`.`{sch.lower()}`")
                schemas_seen.add(sch)
            except Exception as ex:
                print(f"    !! schema {dst_cat}.{sch}: {ex}")
                continue

        objects.append({
            "src_db":  src_db,  "dst_cat": dst_cat,
            "schema":  sch,     "name":    name,
            "is_view": kind.upper() == "VIEW",
        })

    print(f"  {src_db}: {len(rows)} objects discovered")

print(f"\n  {len(objects)} objects to copy")

# COMMAND ----------

# MAGIC %md ### Dependency extraction

# COMMAND ----------

_THREE_PART = re.compile(
    r'(?:"([^"]+)"|([A-Z_][A-Z0-9_]*))'
    r'\.(?:"([^"]+)"|([A-Z_][A-Z0-9_]*))'
    r'\.(?:"([^"]+)"|([A-Z_][A-Z0-9_]*))',
    re.IGNORECASE,
)
_SQL_KEYWORDS = {
    "SELECT","FROM","WHERE","JOIN","ON","AND","OR","NOT","AS","WITH","UNION",
    "GROUP","ORDER","BY","HAVING","LIMIT","DISTINCT","CASE","WHEN","THEN","ELSE","END",
    "LEFT","RIGHT","INNER","OUTER","FULL","CROSS","INTO","INSERT","UPDATE","DELETE",
}

def extract_dependencies(sql):
    refs = set()
    for m in _THREE_PART.finditer(sql or ""):
        db, sch, tbl = (m.group(1) or m.group(2),
                        m.group(3) or m.group(4),
                        m.group(5) or m.group(6))
        if any(p.upper() in _SQL_KEYWORDS for p in (db, sch, tbl)):
            continue
        refs.add((db.upper(), sch.upper(), tbl.upper()))
    return refs

def get_view_definition(src_db, schema, name):
    rows = sf_query(src_db, f"""
        SELECT view_definition
        FROM {src_db}.information_schema.views
        WHERE table_schema = '{schema}' AND table_name = '{name}'
    """).collect()
    if not rows:
        return None
    return rows[0].asDict().get("VIEW_DEFINITION")

# COMMAND ----------

# MAGIC %md ### 6 — dependency pre-flight

# COMMAND ----------

databases_in_run = {p["source_database"].upper() for p in pairs}
view_count       = sum(1 for o in objects if o["is_view"])
external_refs    = []

print(f"  scanning {view_count} view definitions for external dependencies...")

for o in objects:
    if not o["is_view"]:
        continue
    try:
        view_sql = get_view_definition(o["src_db"], o["schema"], o["name"])
    except Exception:
        continue
    if not view_sql:
        continue

    view_id = f"{o['src_db']}.{o['schema']}.{o['name']}"
    for db, sch, tbl in extract_dependencies(view_sql):
        if db not in databases_in_run:
            external_refs.append((view_id, f"{db}.{sch}.{tbl}"))

if external_refs:
    print(f"\n  ⚠  {len(external_refs)} external dependencies found:")
    print(f"     these views reference databases NOT in your CSV.")
    print(f"     they will FAIL unless you add the missing databases.\n")

    by_db = {}
    for view_id, ref in external_refs:
        missing_db = ref.split(".")[0]
        by_db.setdefault(missing_db, []).append((view_id, ref))

    for missing_db, items in sorted(by_db.items()):
        print(f"     missing database: {missing_db}")
        for view_id, ref in items:
            print(f"       {view_id}")
            print(f"         needs → {ref}")
        print()

    print(f"  to fix: add {sorted(by_db)} to your CSV,")
    print(f"          or accept these views will fail and add them manually.\n")
else:
    print(f"  ✓ no external dependencies — all view refs resolve within this run")

# COMMAND ----------

# MAGIC %md ### 7 — copy each object

# COMMAND ----------

def rewrite_view_sql(sql, src_db, dst_cat):
    for src in [src_db, src_db.lower(), src_db.upper()]:
        sql = re.sub(rf'(?<![\w"]){re.escape(src)}\.',
                     f"{dst_cat.lower()}.", sql)
        sql = sql.replace(f'"{src}".', f'"{dst_cat.lower()}".')
    return sql

def copy_object(o):
    src_db,     dst_cat   = o["src_db"], o["dst_cat"]
    sch,        name      = o["schema"], o["name"]
    dst_schema, dst_name  = sch.lower(), name.lower()
    dst_fq    = fq(dst_cat, dst_schema, dst_name)
    dst_plain = plain(dst_cat, dst_schema, dst_name)

    try:
        if o["is_view"]:
            view_sql = get_view_definition(src_db, sch, name)
            if not view_sql:
                raise RuntimeError("view_definition empty in information_schema.views")
            rewritten = rewrite_view_sql(view_sql, src_db, dst_cat)
            spark.sql(f"CREATE OR REPLACE VIEW {dst_fq} AS {rewritten}")
            print(f"  view: {dst_plain}")
            return {"table": dst_plain, "status": "view_created",
                    "rows": None, "error": None}

        df = sf_read_table(src_db, sch, name)
        (df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(dst_plain))

        rows = spark.sql(f"SELECT COUNT(*) AS n FROM {dst_fq}").collect()[0]["n"]
        print(f"  table: {dst_plain}  ({rows:,} rows)")
        return {"table": dst_plain, "status": "table_created",
                "rows": rows, "error": None}

    except Exception as ex:
        err_msg = str(ex)

        if o["is_view"]:
            try:
                view_sql = get_view_definition(src_db, sch, name)
                if view_sql:
                    deps    = extract_dependencies(view_sql)
                    missing = [f"{d}.{s}.{t}" for d, s, t in deps
                               if d not in databases_in_run]
                    if missing:
                        err_msg = (f"missing dependency — view needs "
                                   f"{', '.join(missing)} which is not in this run. "
                                   f"original error: {ex}")
            except Exception:
                pass

        print(f"  !! {dst_plain}: {err_msg}")
        return {"table": dst_plain, "status": "failed",
                "rows": None, "error": err_msg}


results = []
with ThreadPoolExecutor(max_workers=PARALLELISM) as pool:
    futures = {pool.submit(copy_object, o): o for o in objects}
    for fut in as_completed(futures):
        results.append(fut.result())

# COMMAND ----------

# MAGIC %md ### 8 — summary + log

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
    [(r["table"], r["status"], r["rows"], r["error"], RUN_TS) for r in results],
    LOG_SCHEMA,
)
 .write.format("delta").mode("append").option("mergeSchema", "true")
 .saveAsTable(LOG_TABLE))

print(f"\nlog → {LOG_TABLE}")
print(f"done  |  {RUN_TS}")
