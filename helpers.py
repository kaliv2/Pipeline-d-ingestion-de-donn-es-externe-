import os
import sys
import shutil
from glob import glob
import logging
import configparser
from typing import Dict, Any, Optional
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql import SparkSession, DataFrame

def setup_logging() -> logging.Logger:
    """Configure logging for production."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f'matching_{datetime.now().strftime("%Y%m%d")}.log'
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def get_config(filename: str = r"/app/config/config.ini") -> configparser.ConfigParser:
    """Load configuration from file safely."""
    logger = logging.getLogger(__name__)
    try:
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        with open(filename, 'r') as f:
            config.read_file(f)
        logger.info(f"Loaded config from {filename}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise



def get_env(args: list[str]) -> str:
    """Determine environment from args or environment variable."""
    logger = logging.getLogger(__name__)
    default_env_index = 1
    env = os.environ.get("ENV")
    if env:
        logger.info(f"Using ENV variable: {env}")
        return env
    if len(args) < (default_env_index + 1) or args[default_env_index] == "":
        logger.info("No env specified, using default: prd")
        return "prd"
    
    logger.info(f"Using command line arg: {args[0]}")
    return args[default_env_index]


def create_spark_session(name: str) -> SparkSession:
    """Create optimized Spark session."""
    return (
        SparkSession.builder.appName(name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.localShuffleReader.enabled", "true")
        .config("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )

def load_data_tables(spark, config, env, db, table_keys):
    tables = {}
    for key in table_keys:
        path = config.get(env, key)
        if path.endswith(".csv"):
            tables[key] = spark.read.option("header", "true").csv(path)
        else:
            tables[key] = spark.read.option("header", "true").csv(path)
    return tables

def load_data_tables_datalake(
        spark: SparkSession, 
        config: configparser.ConfigParser,
        env: str, db: str,
        table_keys : list[str],
    ) -> Dict[str, Any]:
    """Load all required data tables in a simplified way."""
    
    # Load tables with optional filtering
    tables = {
        key: (
            spark.table(f"{db}.{config.get(env, key)}").filter(F.col("categ").isin(["3"]))
            if key == "orc_personne_host" else
            spark.table(f"{db}.{config.get(env, key)}")
        )
        for key in table_keys
    }

    return tables

def deduplicate_by_latest(
    df: DataFrame, 
    key_col: str, 
    time_col: str, 
    selected_cols: Optional[list[str]] = None
) -> DataFrame:
    """
    Keep only the latest record per key, selecting specific columns.
    """
    if selected_cols is None:
        selected_cols = [key_col, "raison_sociale", "sigle", "rc_and_ct", "ice"]

    # Get latest timestamp per key
    latest = df.groupBy(key_col).agg(F.max(time_col).alias("max_time"))

    # Alias DataFrames to avoid ambiguity
    df_alias = df.alias("df")
    latest_alias = latest.alias("latest")

    # Join back to keep only the latest record
    deduped = (
        df_alias.join(
            F.broadcast(latest_alias),
            (F.col("df." + key_col) == F.col("latest." + key_col)) &
            (F.col("df." + time_col) == F.col("latest.max_time")),
            "inner"
        )
        .select([F.col(f"df.{col}") for col in selected_cols])
        .dropDuplicates()
    )

    return deduped


def save_datasets(df_awb, df_inforisk, df_awb_name, df_ir_name, db_saving):
    """
    Save datasets with database creation fallback
    """
    logger = logging.getLogger(__name__)
    
    try:
        
        # Save datasets
        df_awb.write.mode("overwrite").format("orc").saveAsTable(
            f"{db_saving}.{df_awb_name}"
        )
        df_inforisk.write.mode("overwrite").format("orc").saveAsTable(
            f"{db_saving}.{df_ir_name}"
        )
        
    except Exception as e:
        logger.warning(f"Hive save failed, falling back to local files: {e}")
        
        # Fallback to local files
        output_dir = "/app/storage/processed"
        df_awb.write.mode("overwrite").parquet(f"{output_dir}/{df_awb_name}")
        df_inforisk.write.mode("overwrite").parquet(f"{output_dir}/{df_ir_name}")
        
        logger.info("Datasets saved successfully as local files")

def save_single_csv(df, output_dir, dataset_name):
    """
    Save a PySpark DataFrame as a single CSV file in Spark-readable format
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Define the output path for Spark directory
        csv_path = os.path.join(output_dir, dataset_name)
        
        # Remove existing directory if it exists (before Spark tries to)
        if os.path.exists(csv_path):
            logger.info(f"Removing existing directory: {csv_path}")
            shutil.rmtree(csv_path, ignore_errors=True)
        
        # Write as single partition with Spark directory structure
        logger.info(f"Writing {dataset_name} to: {csv_path}")
        (df.coalesce(1)
         .write
         .mode("overwrite")
         .option("header", "true")
         .csv(csv_path))
        
        logger.info(f"Successfully saved {dataset_name} to {csv_path}")
        
    except Exception as e:
        logger.error(f"Error saving {dataset_name}: {e}")
        # Clean up on error
        if os.path.exists(csv_path):
            shutil.rmtree(csv_path, ignore_errors=True)
        raise


def save_datasets(df_awb, df_inforisk, df_awb_name, df_ir_name):
    """
    Save datasets as single CSV files locally without Hive
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Output directory for CSV files
        output_dir = "/app/storage/processed"
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving datasets as single CSV files to: {output_dir}")
        
        # Save AWB dataset as single CSV file
        save_single_csv(df_awb, output_dir, df_awb_name)
        
        # Save Inforisk dataset as single CSV file
        save_single_csv(df_inforisk, output_dir, df_ir_name)
        
        # Success log
        logger.info(f"Datasets saved successfully as single CSV files")
        
    except Exception as e:
        logger.error(f"Error saving datasets: {e}")
        raise

def relative_similarity(col1, col2):
    """Calculate relative Levenshtein similarity between two columns."""
    return F.when(
        (col1.isNull())
        | (col2.isNull())
        | (F.length(col1) == 0)
        | (F.length(col2) == 0),
        F.lit(0.0),
    ).otherwise(
        1.0 - (F.levenshtein(col1, col2) / F.greatest(F.length(col1), F.length(col2)))
    )
        
def normalize_column(df, col_name, alias_name):
    """
    Normalize column: remove spaces, lowercase, remove non-alphanumeric except specific chars
    """
    normalized_df = df.filter(
        F.col(col_name).isNotNull()
        & (F.trim(F.col(col_name).cast("string")) != "")
        & (F.trim(F.col(col_name).cast("string")) != "null")
    ).withColumn(
        f"{col_name}_normalized",
        F.lower(
            F.regexp_replace(
                F.trim(F.col(col_name).cast("string")),
                r"\s+",  # Remove all spaces
                "",
            )
        ),
    )

    return normalized_df.alias(alias_name)


def validate_datasets(df_awb: DataFrame, df_ir: DataFrame):
    """Validate input datasets before processing."""
    logger = logging.getLogger(__name__)
    
    # Check for required columns
    required_awb = ["numero_personne_host", "raison_sociale", "rc_and_ct","sigle","ice"]
    required_ir = ["bilid", "raison_sociale", "rc_and_ct","sigle","ice"]
    
    missing_awb = [col for col in required_awb if col not in df_awb.columns]
    missing_ir = [col for col in required_ir if col not in df_ir.columns]
    
    if missing_awb or missing_ir:
        raise ValueError(f"Missing columns - AWB: {missing_awb}, IR: {missing_ir}")
    
    # Check for nulls in key columns
    awb_null_count = df_awb.filter(F.col("numero_personne_host").isNull()).count()
    ir_null_count = df_ir.filter(F.col("bilid").isNull()).count()
    
    if awb_null_count > 0 or ir_null_count > 0:
        logger.warning(f"Null keys found - AWB: {awb_null_count}, IR: {ir_null_count}")

    return df_awb, df_ir

def write_tracing_data(
    spark: SparkSession,
    all_candidates_df: DataFrame,
    matched_df: DataFrame,
    trace_table_df: DataFrame,
    db: str,
    trace_table: str,
) -> None:
    """
    Update tracing table with unmatched attempts.
    
    Logic:
    1. Calculate best similarity score per (numero_personne_host, bilid) pair
    2. Exclude already matched hosts
    3. Merge with existing trace data (increment attempts, update max score)
    4. Remove matched hosts from historical data
    5. Atomic table replacement via temp table
    
    Args:
        spark: SparkSession
        all_candidates_df: All candidate pairs with jaccardDist
        matched_df: Successfully matched pairs
        trace_table_df: Existing trace data
        db: Database name
        trace_table: Trace table name
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting optimized trace update...")
        current_time = F.current_timestamp()
        
        # Prepare current attempts (best score per pair)
        current_attempts = (
            all_candidates_df
            .select(
                F.col("numero_personne_host").cast("int"),
                F.col("bilid").cast("int"),
                (1 - F.col("jaccardDist").cast("double")).alias("new_score")
            )
            .groupBy("numero_personne_host", "bilid")
            .agg(F.max("new_score").alias("new_score"))
        )
        
        # Broadcast matched hosts for efficient filtering
        matched_hosts = F.broadcast(
            matched_df.select(F.col("numero_personne_host").cast("int")).distinct()
        )
        
        # Process only new attempts that aren't in existing trace
        new_entries = (
            current_attempts.alias("new")
            .join(trace_table_df.alias("existing"), ["numero_personne_host", "bilid"], "left_anti")
            .join(matched_hosts, "numero_personne_host", "left_anti")
            .select(
                "numero_personne_host",
                "bilid",
                F.lit(1).alias("attempts_count"),  # First attempt
                F.col("new.new_score").alias("similarity_score"),
                current_time.alias("last_attempt_date")
            )
        )

        # Update existing entries that have new attempts
        updated_entries = (
            trace_table_df.alias("existing")
            .join(current_attempts.alias("new"), ["numero_personne_host", "bilid"], "inner")
            .join(matched_hosts, "numero_personne_host", "left_anti")
            .select(
                "numero_personne_host",
                "bilid",
                (F.col("existing.attempts_count") + 1).alias("attempts_count"),  # Increment attempt
                F.greatest(
                    F.col("new.new_score"),
                    F.coalesce(F.col("existing.similarity_score"), F.lit(0.0))
                ).alias("similarity_score"),
                current_time.alias("last_attempt_date")  # Always update timestamp for new attempts
            )
        )

        # Preserve unchanged entries (existing records with no new attempts)
        unchanged_entries = (
            trace_table_df.alias("existing")
            .join(current_attempts.alias("new"), ["numero_personne_host", "bilid"], "left_anti")
            .join(matched_hosts, "numero_personne_host", "left_anti")
            .select(
                "numero_personne_host",
                "bilid",
                "existing.attempts_count",
                F.coalesce(F.col("existing.similarity_score"), F.lit(0.0)).alias("similarity_score"),
                "existing.last_attempt_date"
            )
        )

        # Union all components
        final_trace = new_entries.union(updated_entries).union(unchanged_entries)
        
        # Check for data before writing
        if final_trace.rdd.isEmpty():
            logger.info("No trace data to write.")
            return
        
        # Atomic table swap
        temp_table = f"{trace_table}_temp"
        
        logger.info("Writing trace records to temporary table...")
        
        final_trace.write.mode("overwrite").format("orc").saveAsTable(
            f"{db}.{temp_table}"
        )
        
        spark.sql(f"DROP TABLE IF EXISTS {db}.{trace_table}")
        spark.sql(f"ALTER TABLE {db}.{temp_table} RENAME TO {trace_table}")
        
        logger.info("Trace update complete")
        
    except Exception as e:
        logger.error(f"Trace update failed: {str(e)}", exc_info=True)
        spark.sql(f"DROP TABLE IF EXISTS {db}.{trace_table}_temp")
        raise