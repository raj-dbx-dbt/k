# Databricks notebook source
# DBTITLE 1,Fetch all jobs
from databricks.sdk import WorkspaceClient
from datetime import datetime, timedelta, timezone
import time

w = WorkspaceClient()

# Get the days_lookback parameter
days_lookback = int(dbutils.widgets.get("days_lookback"))
cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_lookback)
cutoff_ms = int(cutoff_time.timestamp() * 1000)

print(f"Looking back {days_lookback} days (since {cutoff_time.strftime('%Y-%m-%d')})")
print("Fetching all jobs...")

# Get all jobs
jobs = list(w.jobs.list())
print(f"Found {len(jobs)} jobs total")

# COMMAND ----------

# DBTITLE 1,Identify jobs on all-purpose clusters with latest run info
from databricks.sdk.service.compute import ClusterDetails

# Build a cache of cluster_id -> cluster details
print("Fetching cluster list for name resolution...")
clusters = list(w.clusters.list())
cluster_details_map = {}
for c in clusters:
    autoscale_min = c.autoscale.min_workers if c.autoscale else None
    autoscale_max = c.autoscale.max_workers if c.autoscale else None
    num_workers = c.num_workers if c.num_workers is not None else None
    
    if autoscale_min is not None:
        workers_info = f"Autoscale {autoscale_min}-{autoscale_max}"
    elif num_workers is not None:
        workers_info = f"Fixed {num_workers}"
    else:
        workers_info = "Unknown"
    
    cluster_details_map[c.cluster_id] = {
        "cluster_name": c.cluster_name,
        "node_type_id": c.node_type_id or "Unknown",
        "driver_node_type_id": c.driver_node_type_id or c.node_type_id or "Unknown",
        "workers_config": workers_info,
        "num_workers": num_workers,
        "autoscale_min": autoscale_min,
        "autoscale_max": autoscale_max,
        "spark_version": c.spark_version or "Unknown",
        "cluster_source": str(c.cluster_source) if c.cluster_source else "Unknown"
    }

print(f"Found {len(cluster_details_map)} clusters")

# Get workspace host for building job URLs
workspace_host = w.config.host.rstrip("/")

# Identify jobs using all-purpose (existing) clusters
results = []

print(f"Inspecting {len(jobs)} jobs for all-purpose cluster usage...")
for i, job in enumerate(jobs):
    job_id = job.job_id
    
    try:
        full_job = w.jobs.get(job_id)
    except Exception as e:
        continue
    
    job_name = full_job.settings.name if full_job.settings else None
    tasks = full_job.settings.tasks if full_job.settings and full_job.settings.tasks else []
    
    existing_cluster_ids = set()
    for task in tasks:
        if task.existing_cluster_id:
            existing_cluster_ids.add(task.existing_cluster_id)
    
    if not existing_cluster_ids:
        continue
    
    # Get the latest run within the lookback window
    try:
        runs = list(w.jobs.list_runs(
            job_id=job_id,
            start_time_from=cutoff_ms,
            limit=1
        ))
    except Exception:
        runs = []
    
    latest_run_id = None
    latest_run_start = None
    latest_run_state = None
    
    if runs:
        latest_run = runs[0]
        latest_run_id = latest_run.run_id
        latest_run_start = datetime.fromtimestamp(latest_run.start_time / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if latest_run.start_time else None
        latest_run_state = str(latest_run.state.result_state) if latest_run.state and latest_run.state.result_state else str(latest_run.state.life_cycle_state) if latest_run.state else None
    
    # Build clickable job URL
    job_url = f"{workspace_host}/jobs/{job_id}"
    
    for cluster_id in existing_cluster_ids:
        details = cluster_details_map.get(cluster_id, {})
        cluster_name = details.get("cluster_name", "UNKNOWN/TERMINATED")
        results.append({
            "job_id": job_id,
            "job_name": job_name,
            "job_url": job_url,
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "worker_node_type": details.get("node_type_id", "Unknown"),
            "driver_node_type": details.get("driver_node_type_id", "Unknown"),
            "workers_config": details.get("workers_config", "Unknown"),
            "autoscale_min_workers": details.get("autoscale_min"),
            "autoscale_max_workers": details.get("autoscale_max"),
            "spark_version": details.get("spark_version", "Unknown"),
            "latest_run_id": latest_run_id,
            "latest_run_start_time": latest_run_start,
            "latest_run_state": latest_run_state,
            "num_tasks_on_cluster": sum(1 for t in tasks if t.existing_cluster_id == cluster_id)
        })

print(f"\nFound {len(results)} job-cluster combinations using all-purpose clusters")

# COMMAND ----------

# DBTITLE 1,Create DataFrame and display results
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType
from pyspark.sql.functions import current_timestamp, col

schema = StructType([
    StructField("job_id", LongType(), True),
    StructField("job_name", StringType(), True),
    StructField("job_url", StringType(), True),
    StructField("cluster_id", StringType(), True),
    StructField("cluster_name", StringType(), True),
    StructField("worker_node_type", StringType(), True),
    StructField("driver_node_type", StringType(), True),
    StructField("workers_config", StringType(), True),
    StructField("autoscale_min_workers", IntegerType(), True),
    StructField("autoscale_max_workers", IntegerType(), True),
    StructField("spark_version", StringType(), True),
    StructField("latest_run_id", LongType(), True),
    StructField("latest_run_start_time", StringType(), True),
    StructField("latest_run_state", StringType(), True),
    StructField("num_tasks_on_cluster", IntegerType(), True)
])

df = spark.createDataFrame(results, schema=schema)
df = df.withColumn("snapshot_timestamp", current_timestamp())

# Attempt to join cost data from system.billing.usage
try:
    cost_df = spark.sql(f"""
        SELECT 
            usage_metadata.cluster_id,
            ROUND(SUM(usage_quantity), 2) AS total_dbus_consumed,
            ROUND(SUM(usage_quantity * list_prices.pricing.default), 2) AS estimated_cost_usd
        FROM system.billing.usage
        LEFT JOIN system.billing.list_prices 
            ON usage.sku_name = list_prices.sku_name
            AND usage.usage_start_time >= list_prices.price_start_time
            AND (list_prices.price_end_time IS NULL OR usage.usage_start_time < list_prices.price_end_time)
        WHERE usage_start_time >= '{cutoff_time.strftime('%Y-%m-%d')}'
            AND usage_metadata.cluster_id IS NOT NULL
        GROUP BY usage_metadata.cluster_id
    """)
    df = df.join(cost_df, df.cluster_id == cost_df.cluster_id, "left").drop(cost_df.cluster_id)
    print("Successfully joined cost data from system.billing.usage")
except Exception as e:
    print(f"Could not pull cost data (system.billing.usage may not be accessible): {e}")
    from pyspark.sql.functions import lit
    df = df.withColumn("total_dbus_consumed", lit(None).cast("double"))
    df = df.withColumn("estimated_cost_usd", lit(None).cast("double"))

# Collect to pandas to avoid recomputation on save (serverless doesn't support cache)
pdf = df.toPandas()
print(f"Total rows: {len(pdf)}")
display(pdf)

# COMMAND ----------

# DBTITLE 1,Save results to Delta table
# Save to Delta table - instant since data is already in memory (pandas)
TARGET_TABLE = "main.default.jobs_allpurpose_cluster_usage"

spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TARGET_TABLE)

print(f"Data saved to {TARGET_TABLE}")