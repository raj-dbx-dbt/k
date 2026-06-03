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
# MAGIC 8. Translates non-SQL procedures via ai_query() for human review
# MAGIC
# MAGIC **First run:** writes a starter CSV and exits — edit it, then re-run.
# MAGIC **Subsequent runs:** does the migration end-to-end.
# MAGIC
# MAGIC References:
# MAGIC - Snowflake Spark connector — https://docs.snowflake.com/en/user-guide/spark-connector-use
# MAGIC - Databricks/Snowflake integration — https://docs.databricks.com/aws/en/connect/external-systems/snowflake
# MAGIC - ai_query() — https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query

# COMMAND ----------

# MAGIC %md ### Config

# COMMAND ----------

# ── snowflake connection ─────────────────────────────────────────────────────

# find the account URL in snowsight under Admin → Accounts.
# format: <account_identifier>.snowflakecomputing.com
SF_URL       = "KESTRA-KESTRAPRD_AZURE_EASTUS2.snowflakecomputing.com"
SF_WAREHOUSE = "xsmall_wh"
SF_ROLE      = "READ_ONLY"

# credentials live in databricks secrets. one-time setup from your terminal:
#   databricks secrets create-scope snowflake --profile prd
#   databricks secrets put-secret snowflake username --profile prd
#   databricks secrets put-secret snowflake password --profile prd
SF_SECRET_SCOPE = "snowflake"

# ── databricks workspace ─────────────────────────────────────────────────────

INVENTORY_CATALOG = "kh_snowringcentraldb"
INVENTORY_SCHEMA  = "dbx_inventory"
INVENTORY_VOLUME  = "migration"
MAPPING_FILENAME  = "databases.csv"

# ── migration behavior ───────────────────────────────────────────────────────

PARALLELISM  = 4                          # concurrent copies
SKIP_SCHEMAS = {"INFORMATION_SCHEMA"}     # snowflake system schemas

# ── AI translation (section 9) ───────────────────────────────────────────────

# pay-per-token endpoint for translating non-SQL procedures.
# confirmed via SHOW SERVING ENDPOINTS in this workspace.
AI_ENDPOINT = "databricks-claude-sonnet-4-6"

# COMMAND ----------

import os
import re
import traceback
import pandas as pd
import requests
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pyspark.sql.types import StructType, StructField, StringType, LongType

LOG_TABLE          = f"{INVENTORY_CATALOG}.{INVENTORY_SCHEMA}.migration_runs"
VOLUME_PATH        = f"/Volumes/{INVENTORY_CATALOG}/{INVENTORY_SCHEMA}/{INVENTORY_VOLUME}"
MAPPING_CSV        = f"{VOLUME_PATH}/{MAPPING_FILENAME}"
RUN_TS             = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
JS_PROC_STAGING    = f"{INVENTORY_CATALOG}.{INVENTORY_SCHEMA}.js_procedures_staging"
JS_PROC_TRANSLATED = f"{INVENTORY_CATALOG}.{INVENTORY_SCHEMA}.js_procedures_translated"

print(f"databricks workspace setup  |  {RUN_TS}\n")

# COMMAND ----------

# MAGIC %md ### 1 — set up inventory schema, volume, log table

# COMMAND ----------

# all idempotent — safe to re-run
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{INVENTORY_CATALOG}`.`{INVENTORY_SCHEMA}`")
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{INVENTORY_CATALOG}`.`{INVENTORY_SCHEMA}`.`{INVENTORY_VOLUME}`")

print(f"  inventory schema : {INVENTORY_CATALOG}.{INVENTORY_SCHEMA}")
print(f"  inventory volume : {VOLUME_PATH}")
print(f"  log table        : {LOG_TABLE}")

# COMMAND ----------

# MAGIC %md ### 2 — ensure a mapping CSV exists

# COMMAND ----------

if not os.path.exists(MAPPING_CSV):
    template = pd.DataFrame([
        {"source_database": "SALES_PROD",   "destination_catalog": "migrated_sales_prod"},
        {"source_database": "FINANCE_PROD", "destination_catalog": "migrated_finance_prod"},
    ])
    template.to_csv(MAPPING_CSV, index=False)
    print(f"\n  starter CSV written to {MAPPING_CSV}")
    print(f"  edit it with your real database list, then re-run this notebook.")
    dbutils.notebook.exit("starter CSV created — edit it and re-run")
else:
    print(f"  mapping csv      : {MAPPING_CSV}")

# COMMAND ----------

# MAGIC %md ### Snowflake connection helpers

# COMMAND ----------

# canonical option names per snowflake's official docs.
# credentials read from databricks secrets, never inlined.
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

# verify credentials + networking before doing any real work. let exceptions
# raise with full trace — no swallowing, no rewriting.

print(f"  testing snowflake connection...")
probe = sf_query("SNOWFLAKE",
    "SELECT CURRENT_VERSION() AS v, CURRENT_USER() AS u, CURRENT_ROLE() AS r"
).collect()[0].asDict()
print(f"  ok — snowflake {probe['V']}  |  user={probe['U']}  role={probe['R']}")

# COMMAND ----------

# show this cluster's egress IP — useful for snowflake network policy allow-listing
print(f"  cluster egress IP: {requests.get('https://api.ipify.org').text}")

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

# one round trip per database — pull tables/views from INFORMATION_SCHEMA.TABLES
# and procedures from INFORMATION_SCHEMA.PROCEDURES. everything lands in one
# `objects` list with a `kind` field that drives the copy logic later.

objects = []
for p in pairs:
    src_db, dst_cat = p["source_database"], p["destination_catalog"]

    table_rows = sf_query(src_db, f"""
        SELECT table_schema, table_name, table_type
        FROM {src_db}.information_schema.tables
        WHERE table_schema NOT IN ('{"','".join(SKIP_SCHEMAS)}')
        ORDER BY table_schema, table_name
    """).collect()

    proc_rows = sf_query(src_db, f"""
        SELECT procedure_schema, procedure_name,
               argument_signature, data_type AS return_type,
               procedure_language, procedure_definition, comment
        FROM {src_db}.information_schema.procedures
        WHERE procedure_schema NOT IN ('{"','".join(SKIP_SCHEMAS)}')
        ORDER BY procedure_schema, procedure_name
    """).collect()

    schemas_seen = set()

    for r in table_rows:
        d   = r.asDict()
        sch = d["TABLE_SCHEMA"]
        if sch not in schemas_seen:
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{dst_cat}`.`{sch.lower()}`")
            schemas_seen.add(sch)
        objects.append({
            "kind":    "view" if d["TABLE_TYPE"].upper() == "VIEW" else "table",
            "src_db":  src_db,  "dst_cat": dst_cat,
            "schema":  sch,     "name":    d["TABLE_NAME"],
        })

    for r in proc_rows:
        d   = r.asDict()
        sch = d["PROCEDURE_SCHEMA"]
        if sch not in schemas_seen:
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{dst_cat}`.`{sch.lower()}`")
            schemas_seen.add(sch)
        objects.append({
            "kind":        "procedure",
            "src_db":      src_db,  "dst_cat": dst_cat,
            "schema":      sch,     "name":    d["PROCEDURE_NAME"],
            "arg_sig":     d.get("ARGUMENT_SIGNATURE") or "()",
            "return_type": d.get("RETURN_TYPE"),
            "language":    (d.get("PROCEDURE_LANGUAGE") or "SQL").upper(),
            "body":        d.get("PROCEDURE_DEFINITION"),
            "comment":     d.get("COMMENT"),
        })

    print(f"  {src_db}: {len(table_rows)} tables/views, {len(proc_rows)} procedures")

print(f"  {len(objects)} objects to copy")

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
view_count       = sum(1 for o in objects if o["kind"] == "view")
external_refs    = []

print(f"  scanning {view_count} view definitions for external dependencies...")

for o in objects:
    if o["kind"] != "view":
        continue
    view_sql = get_view_definition(o["src_db"], o["schema"], o["name"])
    if not view_sql:
        continue

    view_id = f"{o['src_db']}.{o['schema']}.{o['name']}"
    for db, sch, tbl in extract_dependencies(view_sql):
        if db not in databases_in_run:
            external_refs.append((view_id, f"{db}.{sch}.{tbl}"))

if external_refs:
    print(f"\n  {len(external_refs)} external dependencies found:")
    print(f"  these views reference databases NOT in your CSV.")
    print(f"  they will FAIL unless you add the missing databases.\n")

    by_db = {}
    for view_id, ref in external_refs:
        missing_db = ref.split(".")[0]
        by_db.setdefault(missing_db, []).append((view_id, ref))

    for missing_db, items in sorted(by_db.items()):
        print(f"  missing database: {missing_db}")
        for view_id, ref in items:
            print(f"    {view_id}")
            print(f"      needs: {ref}")
        print()

    print(f"  to fix: add {sorted(by_db)} to your CSV,")
    print(f"          or accept these views will fail.\n")
else:
    print(f"  no external dependencies — all view refs resolve within this run")

# COMMAND ----------

# MAGIC %md ### 7 — copy each object

# COMMAND ----------

# ── DBR version check for procedure support ──────────────────────────────────
# spark.version returns the Spark version (e.g. 4.1.0), NOT the DBR version.
# use the DATABRICKS_RUNTIME_VERSION env var to get the actual DBR version.
_dbr_raw = os.environ.get("DATABRICKS_RUNTIME_VERSION", "0.0")
_dbr_match = re.match(r"(\d+)\.(\d+)", _dbr_raw)
can_create_procedures = bool(_dbr_match) and (int(_dbr_match.group(1)), int(_dbr_match.group(2))) >= (17, 0)
print(f"  DBR {_dbr_raw} (Spark {spark.version}) — procedure creation "
      f"{'supported' if can_create_procedures else 'NOT supported (will inventory only)'}")

# ── view DDL helpers ─────────────────────────────────────────────────────────

_VIEW_PREFIX = re.compile(
    r"""
    ^\s* CREATE \s+
    (?: OR \s+ REPLACE \s+ )?
    (?: SECURE \s+ )?
    (?: RECURSIVE \s+ )?
    VIEW \s+
    [\w."]+
    (?: \s* \( [^)]* \) )?
    \s+ AS \b \s*
    """,
    flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

def extract_select_body(view_sql):
    return _VIEW_PREFIX.sub("", view_sql, count=1).strip()

def _replace_func_with_coalesce(sql, func_name, default):
    """Replace FUNC(expr) with COALESCE(expr, default), handling nested parens."""
    pattern = re.compile(rf'\b{func_name}\s*\(', re.IGNORECASE)
    result = []
    i = 0
    while i < len(sql):
        m = pattern.search(sql, i)
        if not m:
            result.append(sql[i:])
            break
        result.append(sql[i:m.start()])
        depth = 1
        j = m.end()
        while j < len(sql) and depth > 0:
            if sql[j] == '(':
                depth += 1
            elif sql[j] == ')':
                depth -= 1
            j += 1
        if depth == 0:
            inner = sql[m.end():j-1]
            result.append(f'COALESCE({inner}, {default})')
            i = j
        else:
            result.append(sql[m.start():m.end()])
            i = m.end()
    return ''.join(result)

def translate_snowflake_functions(sql):
    """Translate Snowflake-specific function signatures to Databricks equivalents."""
    sql = _replace_func_with_coalesce(sql, 'ZEROIFNULL', '0')
    sql = re.sub(
        r'\bREGEXP_REPLACE\s*\(\s*([^,]+)\s*,\s*([^,)]+)\s*\)',
        r"REGEXP_REPLACE(\1, \2, '')",
        sql,
        flags=re.IGNORECASE,
    )
    return sql

def rewrite_sql(sql, src_db, dst_cat):
    """Repoint <src_db>.schema.table to <dst_cat>.schema.table — handles
    upper, lower, and quoted forms. Used by both views and procedures.
    Also translates Snowflake function signatures to Databricks equivalents."""
    for src in [src_db, src_db.lower(), src_db.upper()]:
        sql = re.sub(rf'(?<![\w"]){re.escape(src)}\.',
                     f"{dst_cat.lower()}.", sql)
        sql = sql.replace(f'"{src}".', f'"{dst_cat.lower()}".')
    sql = translate_snowflake_functions(sql)
    return sql

# ── procedure helpers ────────────────────────────────────────────────────────

def translate_procedure_body(body, src_db, dst_cat):
    if not body:
        return body
    body = rewrite_sql(body, src_db, dst_cat)
    body = re.sub(r"\bEXECUTE\s+AS\s+(CALLER|OWNER)\b", "",
                  body, flags=re.IGNORECASE)
    return body

def build_procedure_ddl(o, dst_fq):
    body = translate_procedure_body(o["body"], o["src_db"], o["dst_cat"])
    args = o["arg_sig"] if o["arg_sig"].startswith("(") else f"({o['arg_sig']})"
    comment_clause = ""
    if o.get("comment"):
        safe = o["comment"].replace("'", "''")
        comment_clause = f"COMMENT '{safe}'"
    return f"""
        CREATE OR REPLACE PROCEDURE {dst_fq} {args}
        LANGUAGE SQL
        SQL SECURITY INVOKER
        {comment_clause}
        AS
        {body}
    """

# ── single unified copy function ─────────────────────────────────────────────

def copy_object(o):
    src_db,     dst_cat   = o["src_db"], o["dst_cat"]
    sch,        name      = o["schema"], o["name"]
    dst_schema, dst_name  = sch.lower(), name.lower()
    dst_fq    = fq(dst_cat, dst_schema, dst_name)
    dst_plain = plain(dst_cat, dst_schema, dst_name)

    try:
        if o["kind"] == "table":
            # detect TIME, TIMESTAMP_NTZ, and NUMBER(p,0) for type correction
            col_rows = sf_query(src_db, f"""
                SELECT column_name, data_type, numeric_precision, numeric_scale
                FROM {src_db}.information_schema.columns
                WHERE table_schema = '{sch}' AND table_name = '{name}'
                ORDER BY ordinal_position
            """).collect()

            time_cols = {r.asDict()["COLUMN_NAME"] for r in col_rows
                         if r.asDict()["DATA_TYPE"].upper() == "TIME"}
            ts_ntz_cols = {r.asDict()["COLUMN_NAME"] for r in col_rows
                          if r.asDict()["DATA_TYPE"].upper() in
                          ("TIMESTAMP_NTZ", "TIMESTAMP WITHOUT TIME ZONE")}
            int_cast_cols = {}
            for r in col_rows:
                d = r.asDict()
                if d["DATA_TYPE"].upper() in ("NUMBER", "DECIMAL", "NUMERIC"):
                    scale = int(d.get("NUMERIC_SCALE") or 0)
                    prec  = int(d.get("NUMERIC_PRECISION") or 38)
                    if scale == 0:
                        int_cast_cols[d["COLUMN_NAME"]] = "int" if prec <= 9 else "bigint"

            if time_cols:
                select_exprs = []
                for r in col_rows:
                    c = r.asDict()["COLUMN_NAME"]
                    if c in time_cols:
                        select_exprs.append(
                            f"TO_VARCHAR(\"{c}\", 'HH24:MI:SS') AS \"{c}\"")
                    else:
                        select_exprs.append(f'"{c}"')
                query_sql = f'SELECT {", ".join(select_exprs)} FROM "{sch}"."{name}"'
                df = sf_query(src_db, query_sql)
            else:
                df = sf_read_table(src_db, sch, name)

            from pyspark.sql.functions import col as _col
            cast_map = {}
            if ts_ntz_cols:
                for c in ts_ntz_cols:
                    cast_map[c] = _col(c).cast("timestamp_ntz")
            if int_cast_cols:
                for c, spark_type in int_cast_cols.items():
                    cast_map[c] = _col(c).cast(spark_type)
            if cast_map:
                df = df.withColumns(cast_map)
                try:
                    spark.sql(f"ALTER TABLE {dst_fq} SET TBLPROPERTIES ('delta.feature.timestampNtz' = 'supported')")
                except Exception:
                    pass

            (df.write
                .format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .saveAsTable(dst_plain))
            rows = spark.sql(f"SELECT COUNT(*) AS n FROM {dst_fq}").collect()[0]["n"]
            print(f"  table: {dst_plain}  ({rows:,} rows)")
            if time_cols:
                print(f"         TIME->STRING cols: {sorted(time_cols)}")
            if ts_ntz_cols:
                print(f"         TIMESTAMP_NTZ cols: {sorted(ts_ntz_cols)}")
            if int_cast_cols:
                print(f"         NUMBER->INT/BIGINT cols: {sorted(int_cast_cols.items())}")
            return {"table": dst_plain, "kind": "table",
                    "status": "table_created", "rows": rows, "error": None}

        elif o["kind"] == "view":
            view_sql = get_view_definition(src_db, sch, name)
            if not view_sql:
                raise RuntimeError("view_definition empty in information_schema.views")
            body      = extract_select_body(view_sql)
            rewritten = rewrite_sql(body, src_db, dst_cat)
            spark.sql(f"CREATE OR REPLACE VIEW {dst_fq} AS {rewritten}")
            print(f"  view: {dst_plain}")
            return {"table": dst_plain, "kind": "view",
                    "status": "view_created", "rows": None, "error": None}

        elif o["kind"] == "procedure":
            if not can_create_procedures:
                print(f"  procedure (inventory only): {dst_plain}")
                return {"table": dst_plain, "kind": "procedure",
                        "status": "inventoried_dbr_too_old",
                        "rows": None, "error": None}
            if o["language"] != "SQL":
                print(f"  procedure (manual port, {o['language']}): {dst_plain}")
                return {"table": dst_plain, "kind": "procedure",
                        "status": f"inventoried_manual_port_{o['language'].lower()}",
                        "rows": None, "error": None}
            spark.sql(build_procedure_ddl(o, dst_fq))
            print(f"  procedure: {dst_plain}")
            return {"table": dst_plain, "kind": "procedure",
                    "status": "procedure_created", "rows": None, "error": None}

    except Exception as ex:
        full_trace = traceback.format_exc()
        print(f"\n  !! {dst_plain}\n     {ex}\n\n{full_trace}\n")
        return {"table": dst_plain, "kind": o["kind"],
                "status": "failed", "rows": None, "error": full_trace}


# ── dependency-ordered execution ─────────────────────────────────────────────
# 1. tables + procedures run in parallel (no internal dependencies)
# 2. views are topologically sorted so upstream views are created first
# 3. views at the same depth level run in parallel within that level

non_views = [o for o in objects if o["kind"] != "view"]
views     = [o for o in objects if o["kind"] == "view"]

view_key = lambda o: (o["src_db"].upper(), o["schema"].upper(), o["name"].upper())
view_map = {view_key(v): v for v in views}

deps = defaultdict(set)
for v in views:
    vk = view_key(v)
    view_sql = get_view_definition(v["src_db"], v["schema"], v["name"])
    if not view_sql:
        continue
    for ref in extract_dependencies(view_sql):
        if ref in view_map and ref != vk:
            deps[vk].add(ref)

in_degree = defaultdict(int)
for vk in view_map:
    in_degree[vk] += 0
for vk, dep_set in deps.items():
    in_degree[vk] = len(dep_set)

reverse_deps = defaultdict(set)
for vk, dep_set in deps.items():
    for d in dep_set:
        reverse_deps[d].add(vk)

levels = []
queue = deque([vk for vk, deg in in_degree.items() if deg == 0])
visited = set()
while queue:
    level = list(queue)
    levels.append(level)
    queue = deque()
    for vk in level:
        visited.add(vk)
        for downstream in reverse_deps[vk]:
            in_degree[downstream] -= 1
            if in_degree[downstream] == 0:
                queue.append(downstream)

circular = [vk for vk in view_map if vk not in visited]
if circular:
    levels.append(circular)
    print(f"  {len(circular)} view(s) have circular dependencies — will attempt anyway")

print(f"  execution plan: {len(non_views)} tables/procs (parallel), "
      f"{len(views)} views in {len(levels)} level(s)")
for i, lvl in enumerate(levels):
    print(f"    level {i}: {len(lvl)} view(s)")

results = []
print(f"\n  ── phase 1: tables + procedures ──")
with ThreadPoolExecutor(max_workers=PARALLELISM) as pool:
    futures = {pool.submit(copy_object, o): o for o in non_views}
    for fut in as_completed(futures):
        results.append(fut.result())

print(f"\n  ── phase 2: views (dependency order) ──")
for i, lvl in enumerate(levels):
    lvl_objects = [view_map[vk] for vk in lvl]
    with ThreadPoolExecutor(max_workers=PARALLELISM) as pool:
        futures = {pool.submit(copy_object, o): o for o in lvl_objects}
        for fut in as_completed(futures):
            results.append(fut.result())

# COMMAND ----------

# MAGIC %md ### 8 — summary + log

# COMMAND ----------

tally = lambda s: sum(1 for r in results if r["status"] == s)
kind  = lambda k: sum(1 for r in results if r["kind"]   == k)

print(f"\n{'─'*60}")
print(f"  tables                          : {kind('table')}")
print(f"  views                           : {kind('view')}")
print(f"  procedures (created)            : {tally('procedure_created')}")
print(f"  procedures (manual port needed) : "
      f"{sum(1 for r in results if r['status'].startswith('inventoried_manual_port'))}")
print(f"  procedures (dbr too old)        : {tally('inventoried_dbr_too_old')}")
print(f"  failed                          : {tally('failed')}")
print(f"{'─'*60}")

for r in (r for r in results if r["status"] == "failed"):
    print(f"\n  failed: {r['table']}\n    {r['error']}")

LOG_SCHEMA = StructType([
    StructField("table",       StringType(), True),
    StructField("kind",        StringType(), True),
    StructField("status",      StringType(), True),
    StructField("rows",        LongType(),   True),
    StructField("error",       StringType(), True),
    StructField("migrated_at", StringType(), True),
])

(spark.createDataFrame(
    [(r["table"], r["kind"], r["status"], r["rows"], r["error"], RUN_TS)
     for r in results],
    LOG_SCHEMA,
)
.write.format("delta").mode("append").option("mergeSchema", "true")
.saveAsTable(LOG_TABLE))

print(f"\nlog → {LOG_TABLE}")
print(f"done  |  {RUN_TS}")

# COMMAND ----------

# MAGIC %md ### 9 — AI-assisted translation of non-SQL procedures

# COMMAND ----------

# stage every non-SQL procedure body, then call ai_query() in a single SQL
# pass to translate each one into a databricks SQL procedure.
# output is a human-review starting point — read every translation before
# running it on production.

non_sql_rows = []
for p in pairs:
    src_db  = p["source_database"]
    dst_cat = p["destination_catalog"]

    rows = sf_query(src_db, f"""
        SELECT procedure_schema, procedure_name,
               procedure_language, argument_signature,
               data_type AS return_type,
               procedure_definition, comment
        FROM {src_db}.information_schema.procedures
        WHERE procedure_language != 'SQL'
          AND procedure_schema NOT IN ('{"','".join(SKIP_SCHEMAS)}')
    """).collect()

    for r in rows:
        d = r.asDict()
        non_sql_rows.append((
            f"{src_db}.{d['PROCEDURE_SCHEMA']}.{d['PROCEDURE_NAME']}",
            f"{dst_cat}.{d['PROCEDURE_SCHEMA'].lower()}.{d['PROCEDURE_NAME'].lower()}",
            d["PROCEDURE_LANGUAGE"],
            d.get("ARGUMENT_SIGNATURE") or "()",
            d.get("RETURN_TYPE"),
            d.get("PROCEDURE_DEFINITION"),
            d.get("COMMENT"),
            src_db,
            dst_cat,
        ))

print(f"  {len(non_sql_rows)} non-SQL procedures to translate")

if not non_sql_rows:
    print("  nothing to translate — skipping AI step")
else:
    STAGING_SCHEMA = StructType([
        StructField("source_name",     StringType(), True),
        StructField("target_name",     StringType(), True),
        StructField("language",        StringType(), True),
        StructField("arg_signature",   StringType(), True),
        StructField("return_type",     StringType(), True),
        StructField("original_body",   StringType(), True),
        StructField("comment",         StringType(), True),
        StructField("source_db",       StringType(), True),
        StructField("target_catalog",  StringType(), True),
    ])

    (spark.createDataFrame(non_sql_rows, STAGING_SCHEMA)
        .write.format("delta").mode("overwrite")
        .saveAsTable(JS_PROC_STAGING))

    print(f"  staged to {JS_PROC_STAGING}")

# COMMAND ----------

if non_sql_rows:
    PROMPT = """You are migrating a Snowflake stored procedure to Databricks SQL Scripting.

Output ONLY a valid Databricks CREATE OR REPLACE PROCEDURE statement. No markdown, no code fences, no commentary, no explanation before or after.

Rules:
- LANGUAGE SQL is the only supported procedure language in Databricks
- Required clauses: SQL SECURITY INVOKER, LANGUAGE SQL
- BEGIN ... END block with DECLARE for locals
- Use EXECUTE IMMEDIATE for dynamic SQL
- Use IDENTIFIER(var) when a variable holds a fully-qualified object name
- Use OUT parameters in place of RETURNS VARIANT
- All identifiers lowercase (Databricks UC convention)
- Replace any reference to the source database with the target catalog
- Collapse per-row JavaScript loops into single MERGE INTO or INSERT ... SELECT
  statements wherever the semantics allow — prefer set-based SQL over row-by-row
- Wrap dynamic SQL execution in EXCEPTION handlers to match the original
  try/catch error handling
- If the original returned a VARIANT array of per-row results, return a STRING
  summary via an OUT parameter (e.g. 'N rows merged from X')

Snowflake-to-Databricks function translations:
  LISTAGG(col, sep) WITHIN GROUP (ORDER BY x)  ->  array_join(collect_list(col) WITHIN GROUP (ORDER BY x), sep)
  REGEXP_REPLACE(s, p)                          ->  regexp_replace(s, p, '')
  to_timestamp_ntz(current_timestamp())         ->  current_timestamp()::timestamp_ntz
  IFF(c, a, b)                                  ->  CASE WHEN c THEN a ELSE b END
  ZEROIFNULL(x)                                 ->  COALESCE(x, 0)
  TO_VARCHAR(x)                                 ->  CAST(x AS STRING)

If you cannot translate the procedure faithfully (the logic depends on JavaScript
features with no SQL equivalent), output a CREATE OR REPLACE PROCEDURE that raises
a clear error noting what needs manual port, rather than producing broken SQL.
"""

    prompt_sql_safe = PROMPT.replace("'", "''")

    spark.sql(f"""
        CREATE OR REPLACE TABLE {JS_PROC_TRANSLATED} AS
        SELECT
            source_name,
            target_name,
            language,
            original_body,
            ai_query(
                '{AI_ENDPOINT}',
                CONCAT(
                    '{prompt_sql_safe}',
                    CHAR(10), 'Source database: ', source_db,
                    CHAR(10), 'Target catalog:  ', target_catalog,
                    CHAR(10), 'Target name:     ', target_name,
                    CHAR(10), 'Source language: ', language,
                    CHAR(10), 'Argument signature: ', arg_signature,
                    CHAR(10), 'Return type: ', COALESCE(return_type, 'VARIANT'),
                    CHAR(10), CHAR(10),
                    'Original procedure body:', CHAR(10),
                    original_body, CHAR(10), CHAR(10),
                    'Now output the Databricks SQL procedure:'
                )
            ) AS translated_sql,
            current_timestamp() AS translated_at,
            false                AS applied
        FROM {JS_PROC_STAGING}
    """)

    print(f"  translations written to {JS_PROC_TRANSLATED}")
    print(f"\n  next: review each row's translated_sql, fix what's wrong, apply manually.")
    print(f"  example query:")
    print(f"    SELECT source_name, target_name, translated_sql")
    print(f"    FROM {JS_PROC_TRANSLATED}")
    print(f"    WHERE NOT applied;")
