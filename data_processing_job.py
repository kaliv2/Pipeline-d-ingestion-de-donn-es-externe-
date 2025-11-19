from helpers import (
    setup_logging,
    get_env,
    get_config,
    create_spark_session,
    load_data_tables,
    save_datasets,
    deduplicate_by_latest,
)
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from scripts.data_sourcing.external_data import enrich_clients_dataframe
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from typing import Tuple
import sys
import logging


def filter_matched_records(
    signalitique_info: DataFrame, personne_host: DataFrame, matched_records: DataFrame
) -> Tuple[DataFrame, DataFrame]:
    """
    Remove already matched records from both datasets.
    Returns only unmatched records for processing.
    """
    logger = logging.getLogger(__name__)

    if matched_records.isEmpty():
        logger.info("No previous matches - processing all data")
        return signalitique_info, personne_host

    try:
        # Cache matched IDs for reuse and broadcast (typically small)
        matched_bilid = F.broadcast(matched_records.select("bilid").distinct())
        matched_num_person = F.broadcast(
            matched_records.select("numero_personne_host").distinct()
        )

        # Filter both datasets using anti-joins
        signalitique_filtered = signalitique_info.join(
            matched_bilid, on="bilid", how="left_anti"
        )
        personne_filtered = personne_host.join(
            matched_num_person, on="numero_personne_host", how="left_anti"
        )

        logger.info("Filtered out matched records successfully")
        return signalitique_filtered, personne_filtered

    except Exception as e:
        logger.error(f"Error filtering matched records: {e}")
        return signalitique_info, personne_host


def filter_by_inforisk_recency(
    signalitique_info: DataFrame, personne_host: DataFrame, trace_table: DataFrame
) -> DataFrame:
    """
    Apply conditional filtering based on Inforisk data recency:
    - Recent IR data: process all unmatched AWB records
    - Stale IR data: process only never-attempted AWB records
    """
    logger = logging.getLogger(__name__)

    # Check for current month Inforisk data
    has_recent_data = not signalitique_info.filter(
        (F.month(F.col("date_technique")) == F.month(F.current_date()))
        & (F.year(F.col("date_technique")) == F.year(F.current_date()))
    ).isEmpty()

    if has_recent_data:
        logger.info("Recent IR data found - processing all unmatched records")
        return personne_host

    # No recent IR - filter out previously attempted AWB records
    logger.info("No recent IR data - excluding previously attempted AWB records")
    attempted_ids = F.broadcast(trace_table.select("numero_personne_host").distinct())
    return personne_host.join(attempted_ids, on="numero_personne_host", how="left_anti")


def enrich_with_ice(
    personne_host: DataFrame, identifiant_personne: DataFrame
) -> DataFrame:
    """
    Add ICE (Identifiant Commun de l'Entreprise) to AWB records.
    """
    logger = logging.getLogger(__name__)

    # Filter for ICE identifiers only (code 09) and broadcast small lookup table
    ice_data = F.broadcast(
        identifiant_personne.filter(F.col("code_type_identifiant") == "09")
        .select(
            F.col("numero_personne_host").alias("num_pers"),
            F.col("valeur_identifiant").alias("ice"),
        )
        .dropDuplicates(["num_pers"])
    )

    result = personne_host.join(
        ice_data, personne_host["numero_personne_host"] == ice_data["num_pers"], "left"
    ).drop("num_pers")

    logger.info("ICE information added successfully")
    return result


def prepare_inforisk_data(signalitique_info: DataFrame, villes: DataFrame) -> DataFrame:
    """
    Prepare Inforisk dataset with standardized columns.
    Returns: bilid, raison_sociale, sigle, rc_and_ct, ice
    """
    if signalitique_info.isEmpty():
        return signalitique_info.sparkSession.createDataFrame(
            [],
            "bilid string, raison_sociale string, sigle string, rc_and_ct string, ice string",
        )

    # Broadcast small cities lookup table
    villes_bc = F.broadcast(villes)

    # Join with cities to get tribunal codes
    with_cities = signalitique_info.join(
        villes_bc, signalitique_info["tribunal"] == villes_bc["ville"], "left"
    ).withColumn("tribunal_code", F.coalesce(F.col("code_ville"), F.col("tribunal")))

    # Create standardized columns
    standardized = with_cities.select(
        F.col("bilid"),
        F.trim(F.lower(F.col("denomination"))).alias("raison_sociale"),
        F.trim(F.lower(F.col("sigle"))).alias("sigle"),
        F.lower(
            F.concat(
                F.trim(F.col("rc").cast("string")),
                F.trim(F.col("tribunal_code").cast("string")),
            )
        ).alias("rc_and_ct"),
        F.col("ice"),
        F.col("date_technique"),
    )

    # Keep only most recent record per bilid
    result = deduplicate_by_latest(
        df=standardized, key_col="bilid", time_col="date_technique"
    )

    return result


def prepare_awb_data(spark: SparkSession, personne_host: DataFrame) -> DataFrame:
    """
    Prepare AWB dataset with standardized columns.
    Returns: numero_personne_host, raison_sociale, sigle, rc_and_ct, ice
    """
    if personne_host.isEmpty():
        return spark.createDataFrame(
            [],
            "numero_personne_host string, raison_sociale string, sigle string, rc_and_ct string, ice string",
        )

    # Create standardized columns
    standardized = personne_host.select(
        F.col("numero_personne_host"),
        F.lower(
            F.concat(
                F.trim(F.col("raisoc1_nom").cast("string")),
                F.trim(F.col("raisoc2_prenom").cast("string")),
            )
        ).alias("raison_sociale"),
        F.trim(F.lower(F.col("sigle"))).alias("sigle"),
        F.lower(
            F.concat(
                F.trim(F.col("no_rgcom").cast("string")),
                F.trim(F.col("centr_imm").cast("string")),
            )
        ).alias("rc_and_ct"),
        F.col("ice"),
        F.col("time"),
    )

    # Keep only most recent record per numero_personne_host
    result = deduplicate_by_latest(
        df=standardized, key_col="numero_personne_host", time_col="time"
    )
    return result


def main():
    """Main data preparation pipeline."""
    logger = setup_logging()
    spark = None

    try:
        logger.info("Starting data preparation process")

        # Initialize environment
        config = get_config()
        env = get_env(sys.argv)
        db = config.get(env, "db")
        db_saving = config.get(env, "db_saving")

        logger.info(f"Environment: {env}, Database: {db}")
        df_awb_name = config.get(env, "dataset_awb")
        df_ir_name = config.get(env, "dataset_ir")
        # Create Spark session
        spark = create_spark_session("DataPreparationJob")

        # Load all required tables
        logger.info("Loading data tables")
        table_keys = [
            "orc_signalitique_info",
            "orc_identifiant_personne",
            "orc_villes",
            "orc_personne_host",
            # "matching_table",
            # "trace_table",
        ]
        tables = load_data_tables(spark, config, env, db, table_keys)

        # Step 1: Filter out already matched records
        # logger.info("Filtering matched records")
        # signalitique_info, personne_host = filter_matched_records(
        #     tables["orc_signalitique_info"],
        #     tables["orc_personne_host"],
        #     tables["matching_table"],
        # )

        # Step 2: Apply recency-based filtering for AWB
        # logger.info("Applying recency-based filtering")
        # personne_host = filter_by_inforisk_recency(
        #     signalitique_info, personne_host, tables["trace_table"]
        # )
        signalitique_info = tables["orc_signalitique_info"]
        personne_host = tables["orc_personne_host"]

        # Step 3: Enrich AWB data with ICE
        logger.info("Enriching AWB data with ICE")
        personne_host = enrich_with_ice(
            personne_host, tables["orc_identifiant_personne"]
        )
        # Cache after enrichment as it will be used multiple times
        personne_host = personne_host.cache()

        # Step 4: Prepare Inforisk dataset
        logger.info("Preparing Inforisk dataset")
        df_inforisk = prepare_inforisk_data(signalitique_info, tables["orc_villes"])

        # Step 5: Prepare AWB dataset
        logger.info("Preparing AWB dataset")
        df_awb = prepare_awb_data(spark, personne_host)

        # Data Sourcng: Enrich AWB data with external information
        logger.info("Enriching AWB dataset with external data")
        df_clients_pd = df_awb.toPandas()
        enriched_pd = enrich_clients_dataframe(df_clients_pd)

        # Reconvertir vers Spark
        df_awb_enriched = spark.createDataFrame(enriched_pd)
        
        # Step 6: Save results
        logger.info("Saving datasets")
        # save_datasets(spark,df_awb, df_inforisk, df_awb_name, df_ir_name, db_saving)
        df_awb_name = "df_awb_out"
        df_ir_name = "df_inforisk_out"
        save_datasets(df_awb_enriched, df_inforisk, df_awb_name, df_ir_name)

        logger.info("Data preparation completed successfully")

    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)
        raise

    finally:
        if spark is not None:
            spark.catalog.clearCache()
            spark.stop()


if __name__ == "__main__":
    main()
