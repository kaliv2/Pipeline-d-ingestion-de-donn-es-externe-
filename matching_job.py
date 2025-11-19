import sys
import logging
from dataclasses import dataclass, field
from typing import Dict
from helpers import (
    setup_logging,
    get_config,
    get_env,
    create_spark_session,
    load_data_tables,
    validate_datasets,
    normalize_column,
    relative_similarity,
    write_tracing_data,
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    HashingTF,
    MinHashLSH,
    NGram,
    RegexTokenizer,
    SQLTransformer,
    StopWordsRemover,
    Tokenizer,
)
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class MatchingConfig:
    """Configuration for matching parameters."""

    lsh_num_hash_tables: int = 3
    hashing_num_features: int = 262144
    min_token_length: int = 1
    lsh_distance_threshold: float = 0.5
    max_candidates_per_record: int = 10
    similarity_threshold: float = 0.65
    weights: Dict[str, float] = field(
        default_factory=lambda: {"name": 0.45, "rc": 0.2, "sigle": 0.05, "ice": 0.3}
    )


# ============================================================================
# LSH Matching Pipeline
# ============================================================================


class LSHMatcher:
    """Handles LSH-based fuzzy matching."""

    def __init__(self, config: MatchingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _build_feature_pipeline(self, company_col: str) -> Pipeline:
        """Build text processing and feature extraction pipeline."""
        return Pipeline(
            stages=[
                SQLTransformer(
                    statement=f"SELECT *, lower(trim({company_col})) AS lower FROM __THIS__"
                ),
                SQLTransformer(
                    statement="""
                SELECT *, translate(lower,
                    'àáâãäåāăąèéêëēĕėęěìíîïĩīĭòóôõöōŏőùúûüũūŭůÿýçćčđñńšž',
                    'aaaaaaaaaeeeeeeeeeeiiiiiiiooooooooouuuuuuuyycccdnnsz'
                ) AS normalized FROM __THIS__
                """
                ),
                Tokenizer(inputCol="normalized", outputCol="token"),
                StopWordsRemover(inputCol="token", outputCol="stop"),
                SQLTransformer(
                    statement="SELECT *, concat_ws(' ', stop) AS concat FROM __THIS__"
                ),
                SQLTransformer(
                    statement="SELECT *, regexp_replace(concat, '[^a-z0-9]', '') AS cleaned FROM __THIS__"
                ),
                RegexTokenizer(
                    pattern="",
                    inputCol="cleaned",
                    outputCol="char",
                    minTokenLength=self.config.min_token_length,
                ),
                NGram(n=2, inputCol="char", outputCol="ngram"),
                HashingTF(
                    inputCol="ngram",
                    outputCol="vector",
                    numFeatures=self.config.hashing_num_features,
                ),
            ]
        )

    def find_candidates(
        self, df_awb: DataFrame, df_ir: DataFrame, company_col: str
    ) -> DataFrame:
        """Find candidate matches using MinHash LSH."""
        self.logger.info("Building LSH feature vectors...")

        # Build features
        pipeline = self._build_feature_pipeline(company_col)
        transformer = pipeline.fit(df_awb)

        df_awb_features = transformer.transform(df_awb).filter(F.size("ngram") > 0)
        df_ir_features = transformer.transform(df_ir).filter(F.size("ngram") > 0)

        # Apply MinHash LSH
        lsh_model = MinHashLSH(
            inputCol="vector",
            outputCol="lsh",
            numHashTables=self.config.lsh_num_hash_tables,
        ).fit(df_awb_features)

        similarity_df = lsh_model.approxSimilarityJoin(
            df_awb_features,
            df_ir_features,
            self.config.lsh_distance_threshold,
            distCol="jaccardDist",
        )

        # Keep top N candidates per AWB record
        window = Window.partitionBy(f"datasetA.{df_awb.columns[0]}").orderBy(
            "jaccardDist"
        )

        return (
            similarity_df.withColumn("rank", F.row_number().over(window))
            .filter(F.col("rank") <= self.config.max_candidates_per_record)
            .drop("rank")
        )


# ============================================================================
# Scoring & Match Selection
# ============================================================================


class MatchScorer:
    """Handles candidate scoring and best match selection."""

    def __init__(self, config: MatchingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def score_candidates(
        self, candidates: DataFrame, rc_col: str, sigle_col: str, ice_col: str
    ) -> DataFrame:
        """Calculate weighted similarity scores for candidates."""
        self.logger.info("Calculating similarity scores...")

        return (
            candidates
            # Calculate similarities once
            .withColumn(
                "sim_rc",
                relative_similarity(F.col(f"{rc_col}_awb"), F.col(f"{rc_col}_ir")),
            )
            .withColumn(
                "sim_sigle",
                relative_similarity(
                    F.col(f"{sigle_col}_awb"), F.col(f"{sigle_col}_ir")
                ),
            )
            .withColumn(
                "sim_ice",
                relative_similarity(F.col(f"{ice_col}_awb"), F.col(f"{ice_col}_ir")),
            )
            # Calculate weighted scores
            .withColumn(
                "score_rc",
                F.when(
                    (F.col(f"{rc_col}_awb").isNotNull())
                    & (F.col(f"{rc_col}_ir").isNotNull())
                    & (F.col("sim_rc") > self.config.similarity_threshold),
                    F.col("sim_rc") * self.config.weights["rc"],
                ).otherwise(0.0),
            )
            .withColumn(
                "score_sigle",
                F.when(
                    (F.col(f"{sigle_col}_awb").isNotNull())
                    & (F.col(f"{sigle_col}_ir").isNotNull())
                    & (F.col("sim_sigle") > self.config.similarity_threshold),
                    F.col("sim_sigle") * self.config.weights["sigle"],
                ).otherwise(0.0),
            )
            .withColumn(
                "score_ice",
                F.when(
                    (F.col(f"{ice_col}_awb").isNotNull())
                    & (F.col(f"{ice_col}_ir").isNotNull())
                    & (F.col("sim_ice") > self.config.similarity_threshold),
                    F.col("sim_ice") * self.config.weights["ice"],
                ).otherwise(0.0),
            )
            .withColumn(
                "score_name", (1.0 - F.col("jaccardDist")) * self.config.weights["name"]
            )
            # Total score
            .withColumn(
                "score",
                F.col("score_name")
                + F.col("score_rc")
                + F.col("score_sigle")
                + F.col("score_ice"),
            )
            .drop("sim_rc", "sim_sigle", "sim_ice")
        )

    def select_best_matches(self, scored: DataFrame) -> DataFrame:
        """Select best match per AWB record, then deduplicate by IR bilid."""
        self.logger.info("Selecting best matches...")

        # Best match per AWB record
        awb_window = Window.partitionBy("numero_personne_host").orderBy(
            F.col("score").desc(), F.col("jaccardDist").asc()
        )
        best_per_awb = (
            scored.withColumn("rank", F.row_number().over(awb_window))
            .filter(F.col("rank") == 1)
            .drop("rank")
        )

        # Deduplicate by IR bilid
        bilid_window = Window.partitionBy("bilid").orderBy(F.col("score").desc())

        return (
            best_per_awb.withColumn("bilid_rank", F.row_number().over(bilid_window))
            .filter(F.col("bilid_rank") == 1)
            .drop("bilid_rank")
            .select(
                "numero_personne_host",
                "bilid",
                "score",
                "jaccardDist",
                "rc_and_ct_awb",
                "rc_and_ct_ir",
                "sigle_awb",
                "sigle_ir",
                "ice_awb",
                "ice_ir",
                "rs_awb",
                "rs_ir",
                F.current_timestamp().alias("match_time"),
            )
        )


# ============================================================================
# Matching Functions
# ============================================================================


def exact_match(df_awb: DataFrame, df_ir: DataFrame, col_name: str) -> DataFrame:
    """Perform exact matching on normalized column."""
    logger = logging.getLogger(__name__)
    logger.info(f"Performing exact match on {col_name}...")

    df_awb_norm = normalize_column(df_awb, col_name, "awb")
    df_ir_norm = normalize_column(df_ir, col_name, "ir")

    return df_awb_norm.join(
        df_ir_norm,
        F.col(f"awb.{col_name}_normalized") == F.col(f"ir.{col_name}_normalized"),
        "inner",
    ).select(
        F.col("awb.numero_personne_host"),
        F.col("ir.bilid"),
        F.lit(1.0).alias("score"),
        F.lit(0.0).alias("jaccardDist"),
        F.col("awb.rc_and_ct").alias("rc_and_ct_awb"),
        F.col("ir.rc_and_ct").alias("rc_and_ct_ir"),
        F.col("awb.sigle").alias("sigle_awb"),
        F.col("ir.sigle").alias("sigle_ir"),
        F.col("awb.ice").alias("ice_awb"),
        F.col("ir.ice").cast("string").alias("ice_ir"),
        F.col("awb.raison_sociale").alias("rs_awb"),
        F.col("ir.raison_sociale").alias("rs_ir"),
        F.current_timestamp().alias("match_time"),
    )


def fuzzy_match(
    df_awb: DataFrame, df_ir: DataFrame, config: MatchingConfig
) -> DataFrame:
    """Perform LSH-based fuzzy matching with scoring."""
    logger = logging.getLogger(__name__)
    logger.info("Performing LSH fuzzy matching...")

    company_col = "raison_sociale"
    rc_col, sigle_col, ice_col = "rc_and_ct", "sigle", "ice"

    # Prepare minimal datasets for LSH
    df_awb_prep = df_awb.select("numero_personne_host", company_col).filter(
        F.col(company_col).isNotNull()
    )

    df_ir_prep = df_ir.select("bilid", company_col).filter(
        F.col(company_col).isNotNull()
    )

    # Find LSH candidates
    matcher = LSHMatcher(config)
    candidates = matcher.find_candidates(df_awb_prep, df_ir_prep, company_col)

    # Join with full records for scoring
    df_awb_cols = df_awb.select(
        "numero_personne_host",
        F.col(rc_col).alias(f"{rc_col}_awb"),
        F.col(sigle_col).alias(f"{sigle_col}_awb"),
        F.col(ice_col).alias(f"{ice_col}_awb"),
    )

    df_ir_cols = df_ir.select(
        "bilid",
        F.col(rc_col).cast("string").alias(f"{rc_col}_ir"),
        F.col(sigle_col).alias(f"{sigle_col}_ir"),
        F.col(ice_col).cast("string").alias(f"{ice_col}_ir"),
    )

    candidates_full = (
        candidates.join(
            df_awb_cols,
            candidates["datasetA.numero_personne_host"]
            == df_awb_cols["numero_personne_host"],
        )
        .join(df_ir_cols, candidates["datasetB.bilid"] == df_ir_cols["bilid"])
        .select(
            F.col("datasetA.numero_personne_host"),
            F.col("datasetB.bilid"),
            F.col(f"datasetA.{company_col}").alias("rs_awb"),
            F.col(f"datasetB.{company_col}").alias("rs_ir"),
            f"{rc_col}_awb",
            f"{rc_col}_ir",
            f"{sigle_col}_awb",
            f"{sigle_col}_ir",
            f"{ice_col}_awb",
            f"{ice_col}_ir",
            "jaccardDist",
        )
    )

    # Score and select best matches
    scorer = MatchScorer(config)
    scored = scorer.score_candidates(candidates_full, rc_col, sigle_col, ice_col)
    best_matches = scorer.select_best_matches(scored)
    return best_matches, candidates_full


def match_datasets(
    df_awb: DataFrame, df_ir: DataFrame, config: MatchingConfig
) -> DataFrame:
    """
    Execute full matching pipeline:
    1. Exact ICE match
    2. Exact RC match on remaining
    3. Fuzzy LSH match on remaining
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting multi-stage matching pipeline...")

    # Stage 1: ICE exact match
    ice_matches = exact_match(df_awb, df_ir, "ice")

    awb_matched_1 = F.broadcast(ice_matches.select("numero_personne_host").distinct())
    ir_matched_1 = F.broadcast(ice_matches.select("bilid").distinct())

    df_awb_rem_1 = df_awb.join(
        awb_matched_1, "numero_personne_host", "left_anti"
    ).cache()
    df_ir_rem_1 = df_ir.join(ir_matched_1, "bilid", "left_anti").cache()

    logger.info("Stage 1 complete based on ICE matching")

    # Stage 2: RC exact match
    rc_matches = exact_match(df_awb_rem_1, df_ir_rem_1, "rc_and_ct")

    awb_matched_2 = F.broadcast(rc_matches.select("numero_personne_host").distinct())
    ir_matched_2 = F.broadcast(rc_matches.select("bilid").distinct())

    df_awb_rem_2 = df_awb_rem_1.join(
        awb_matched_2, "numero_personne_host", "left_anti"
    ).cache()
    df_ir_rem_2 = df_ir_rem_1.join(ir_matched_2, "bilid", "left_anti").cache()

    df_awb_rem_1.unpersist()
    df_ir_rem_1.unpersist()

    logger.info("Stage 2 complete based on RC matching")

    # Stage 3: LSH fuzzy match
    lsh_matches, candidates = fuzzy_match(df_awb_rem_2, df_ir_rem_2, config)

    df_awb_rem_2.unpersist()
    df_ir_rem_2.unpersist()

    logger.info("Stage 3 complete based on LSH fuzzy matching")

    # Combine all matches
    all_matches = ice_matches.unionByName(rc_matches).unionByName(lsh_matches)
    logger.info("Matching pipeline completed successfully")

    return all_matches, candidates


# ============================================================================
# Main
# ============================================================================


def main():
    """Execute matching pipeline."""
    logger = setup_logging()
    spark = None
    df_awb = None
    df_ir = None
    try:
        logger.info("=" * 60)
        logger.info("Starting Entity Matching Pipeline")
        logger.info("=" * 60)

        # Initialize configuration and environment
        config = get_config()
        env = get_env(sys.argv)
        db = config.get(env, "db")
        db_saving = config.get(env, "db_saving")
        # trace_table_name = config.get(env, "trace_table")
        # matching_table_name = config.get(env, "matching_table")

        logger.info(f"Environment: {env}, Database: {db}")
        matching_config = MatchingConfig()

        # Initialize Spark
        spark = create_spark_session("MatchingJob")
        # table_keys = ["dataset_awb", "dataset_ir", "trace_table"]
        table_keys = ["dataset_awb", "dataset_ir"]
        # Load data
        tables = load_data_tables(spark, config, env, db, table_keys)
        # df_awb, df_ir, trace_table = (
        #     tables["dataset_awb"],
        #     tables["dataset_ir"],
        #     tables["trace_table"],
        # )
        df_awb, df_ir = (
            tables["dataset_awb"],
            tables["dataset_ir"],
        )

        # df_awb = tables["dataset_awb"].filter(
        #     ~F.col("raison_sociale").like("%specimen raison_socia2%")
        # )        
        # df_awb, df_ir = validate_datasets(tables["dataset_awb"], tables["dataset_ir"])
        df_awb = df_awb.cache()
        df_ir = df_ir.cache()
        
        # Execute matching
        matches, all_candidates = match_datasets(df_awb, df_ir, matching_config)
        # Save results as CSV
        output_dir = "/app/storage/matched"
        
        logger.info(f"Saving matching results as CSV to: {output_dir}")
        
        # Save AWB dataset as CSV
        matches.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_dir)

        # In DataLake env:
        # Save results
        # matches.write.format("orc").mode("append").insertInto(f"{db_saving}.{matching_table_name}")
        # Update tracing table
        # write_tracing_data(
        #     spark, all_candidates, matches, trace_table, db_saving, trace_table_name
        # )
        logger.info("=" * 60)
        logger.info("Matching pipeline completed successfully")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Explicit cleanup
        if df_awb is not None:
            try:
                df_awb.unpersist()
            except Exception as e:
                logger.warning(f"Failed to unpersist df_awb: {e}")
        
        if df_ir is not None:
            try:
                df_ir.unpersist()
            except Exception as e:
                logger.warning(f"Failed to unpersist df_ir: {e}")
        
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
