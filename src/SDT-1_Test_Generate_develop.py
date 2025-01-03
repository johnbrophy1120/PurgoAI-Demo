# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, sum as spark_sum, when, first
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, TimestampType
from pyspark.sql.window import Window

# Initialize Spark session
spark = SparkSession.builder.appName("PopulateFEvents").getOrCreate()

# Define schema for purgo_poc.all_events
schema = StructType([
    StructField("event_date", StringType(), True),
    StructField("event_name", StringType(), True),
    StructField("event_timestamp", LongType(), True),
    StructField("user_pseudo_id", StringType(), True),
    StructField("event_bundle_sequence_id", IntegerType(), True),
    StructField("event_params", StructType([
        StructField("key", StringType(), True),
        StructField("value", StructType([
            StructField("int_value", IntegerType(), True),
            StructField("string_value", StringType(), True)
        ]), True)
    ]), True),
    StructField("device", StructType([
        StructField("web_info", StructType([
            StructField("browser", StringType(), True)
        ]), True),
        StructField("category", StringType(), True)
    ]), True),
    StructField("geo", StructType([
        StructField("city", StringType(), True),
        StructField("country", StringType(), True),
        StructField("region", StringType(), True)
    ]), True),
    StructField("traffic_source", StructType([
        StructField("source", StringType(), True),
        StructField("medium", StringType(), True)
    ]), True)
])

# Load data from purgo_poc.all_events
all_events_df = spark.read.schema(schema).table("purgo_poc.all_events")

# Transform event_timestamp from unixtime to datetime
all_events_df = all_events_df.withColumn("event_ts", from_unixtime(col("event_timestamp")).cast(TimestampType()))

# Extract session_id and session_number
all_events_df = all_events_df.withColumn("session_id", when(col("event_params.key") == "ga_session_id", col("event_params.value.int_value")))
all_events_df = all_events_df.withColumn("session_number", when(col("event_params.key") == "ga_session_number", col("event_params.value.int_value")))

# Extract page_title, adcontent, campaign, and other string values
all_events_df = all_events_df.withColumn("page_title", when(col("event_params.key") == "page_title", col("event_params.value.string_value")))
all_events_df = all_events_df.withColumn("adcontent", when(col("event_params.key") == "content", col("event_params.value.string_value")))
all_events_df = all_events_df.withColumn("campaign", when(col("event_params.key") == "campaign", col("event_params.value.string_value")))

# Calculate engagement_time
engagement_time_df = all_events_df.filter(col("event_params.key") == "engagement_time_msec").groupBy("user_pseudo_id").agg(spark_sum("event_params.value.int_value").alias("engagement_time"))

# Join engagement_time back to the main DataFrame
all_events_df = all_events_df.join(engagement_time_df, on="user_pseudo_id", how="left")

# Select and rename columns to match f_events schema
f_events_df = all_events_df.select(
    col("event_date").alias("date"),
    col("event_name"),
    col("event_ts"),
    col("user_pseudo_id"),
    col("event_bundle_sequence_id"),
    first("session_id").over(Window.partitionBy("user_pseudo_id")).alias("session_id"),
    first("session_number").over(Window.partitionBy("user_pseudo_id")).alias("session_number"),
    col("device.web_info.browser").alias("device_browser"),
    col("device.category").alias("device_category"),
    col("geo.city").alias("city"),
    col("geo.country").alias("country"),
    col("geo.region").alias("region"),
    first("page_title").over(Window.partitionBy("user_pseudo_id")).alias("page_title"),
    first("adcontent").over(Window.partitionBy("user_pseudo_id")).alias("adcontent"),
    first("campaign").over(Window.partitionBy("user_pseudo_id")).alias("campaign"),
    col("traffic_source.source").alias("traffic_source"),
    col("traffic_source.medium").alias("traffic_medium"),
    first("referral_path").over(Window.partitionBy("user_pseudo_id")).alias("referral_path"),
    first("keyword").over(Window.partitionBy("user_pseudo_id")).alias("keyword"),
    first("session_engaged").over(Window.partitionBy("user_pseudo_id")).alias("session_engaged"),
    first("content").over(Window.partitionBy("user_pseudo_id")).alias("content"),
    first("search_query").over(Window.partitionBy("user_pseudo_id")).alias("search_query"),
    first("form_name").over(Window.partitionBy("user_pseudo_id")).alias("form_name"),
    first("navigation_item_type").over(Window.partitionBy("user_pseudo_id")).alias("navigation_item_type"),
    first("navigation_item_name").over(Window.partitionBy("user_pseudo_id")).alias("navigation_item_name"),
    first("content_name").over(Window.partitionBy("user_pseudo_id")).alias("content_name"),
    col("engagement_time")
)

# Load existing data from f_events
existing_f_events_df = spark.table("purgo_poc.f_events")

# Find missing rows
missing_rows_df = f_events_df.join(existing_f_events_df, on=["date", "event_name", "event_ts"], how="left_anti")

# Write missing rows to f_events
missing_rows_df.write.insertInto("purgo_poc.f_events", overwrite=False)
