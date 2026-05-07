from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import random
from datetime import datetime, timedelta

spark = SparkSession.builder \
    .appName("Energy Analytics") \
    .getOrCreate()

# ==================================
# GENERATE DATA
# ==================================

sectors = ["Industrial_A", "Industrial_B", "Residential_C"]

start_time = datetime.now()

data = []

for i in range(150):

    current_time = start_time + timedelta(minutes=i)

    for sector in sectors:

        power_usage = random.randint(100, 1000)

        data.append((
            current_time.strftime("%Y-%m-%d %H:%M:%S"),
            sector,
            power_usage
        ))

columns = ["timestamp", "sector", "power_usage"]

df = spark.createDataFrame(data, columns)

df = df.withColumn(
    "timestamp",
    to_timestamp(col("timestamp"))
)

# ==================================
# TOTAL ENERGI
# ==================================

energy_total = df.groupBy("sector").agg(
    sum("power_usage").alias("total_power")
)

# ==================================
# AGREGASI 10 MENIT
# ==================================

energy_time = df.withColumn(
    "minute_group",
    floor(minute(col("timestamp")) / 10) * 10
)

energy_time = energy_time.groupBy(
    hour(col("timestamp")).alias("hour"),
    "minute_group",
    "sector"
).agg(
    avg("power_usage").alias("avg_power_usage")
)

# ==================================
# DATA MACHINE LEARNING
# ==================================

ml_energy = df.withColumn(
    "hour",
    hour(col("timestamp"))
).select(
    "hour",
    "power_usage"
)

# ==================================
# ABSOLUTE PATH
# ==================================

base_path = "/home/hp/bigdata-project/uts-tbg-230104040206/output"

# ==================================
# SAVE PARQUET
# ==================================

energy_total.write.mode("overwrite").parquet(
    f"{base_path}/energy_total"
)

energy_time.write.mode("overwrite").parquet(
    f"{base_path}/energy_time"
)

ml_energy.write.mode("overwrite").parquet(
    f"{base_path}/ml_energy"
)

print("=" * 50)
print("PARQUET BERHASIL DIBUAT")
print("=" * 50)

spark.stop()