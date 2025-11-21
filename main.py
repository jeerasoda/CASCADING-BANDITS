from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Window
import numpy as np
import logging
from datetime import datetime
from decimal import Decimal

# ============================================================================
# SETUP LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DEFINE UDFs FOR MATRIX OPERATIONS
# ============================================================================

# UDF to calculate theta = A_I^(-1) * b_i
@udf(ArrayType(DoubleType()))
def calculate_theta(A_I, b_i):
    try:
        # A_I comes as 2D array (6x6)
        A_I_flat = [item for sublist in A_I for item in sublist]
        A_I_array = np.array(A_I_flat, dtype=float).reshape(6, 6)  # 6x6 now
        b_i_array = np.array(b_i, dtype=float)  # length 6
        A_I_inv = np.linalg.inv(A_I_array)
        theta = A_I_inv @ b_i_array
        return theta.tolist()
    except Exception as e:
        logger.error(f"Error calculating theta: {str(e)}")
        return [0.0] * 6

# UDF to calculate raw score = theta^T * x (dot product)
@udf(DoubleType())
def calculate_raw_score(theta, features):
    try:
        theta_array = np.array(theta, dtype=float)  # length 6
        x_array = np.array(features, dtype=float)  # length 6
        raw_score = np.dot(theta_array, x_array)
        return float(raw_score)
    except Exception as e:
        logger.error(f"Error calculating raw score: {str(e)}")
        return 0.0

# UDF to calculate uncertainty = alpha * sqrt(x^T * A_I^(-1) * x)
@udf(DoubleType())
def calculate_uncertainty(A_I, features, alpha=1.0):
    try:
        # A_I comes as 2D array (6x6)
        A_I_flat = [item for sublist in A_I for item in sublist]
        A_I_array = np.array(A_I_flat, dtype=float).reshape(6, 6)  # 6x6
        x_array = np.array(features, dtype=float)  # length 6
        
        A_I_inv = np.linalg.inv(A_I_array)
        uncertainty = alpha * np.sqrt(x_array.T @ A_I_inv @ x_array)
        return float(uncertainty)
    except Exception as e:
        logger.error(f"Error calculating uncertainty: {str(e)}")
        return 0.0

# UDF to update A_I = A_I + x * x^T - Returns 2D array (6x6)
@udf(ArrayType(ArrayType(DecimalType(10, 6))))
def update_A_I(A_I, features):
    try:
        # A_I comes as 2D array (6x6)
        A_I_flat = [item for sublist in A_I for item in sublist]
        A_I_array = np.array(A_I_flat, dtype=float).reshape(6, 6)  # 6x6
        x = np.array(features, dtype=float)  # length 6
        A_I_new = A_I_array + np.outer(x, x)  # 6x6 + 6x6
        # Convert to 2D list with Decimal type
        return [[Decimal(str(round(float(val), 6))) for val in row] for row in A_I_new]
    except Exception as e:
        logger.error(f"Error updating A_I: {str(e)}")
        return A_I

# UDF to update b_i = b_i + x * reward - Returns 1D array (length 6)
@udf(ArrayType(DecimalType(10, 6)))
def update_b_i(b_i, features, reward):
    try:
        b_i_array = np.array(b_i, dtype=float)  # length 6
        x = np.array(features, dtype=float)  # length 6
        b_i_new = b_i_array + x * reward  # length 6
        # Convert to list with Decimal type
        return [Decimal(str(round(float(val), 6))) for val in b_i_new]
    except Exception as e:
        logger.error(f"Error updating b_i: {str(e)}")
        return b_i
    
    
# ============================================================================
# PART 1: RE-RANKING PRODUCTS (PREDICTION PHASE) - CORRECTED
# ============================================================================

logger.info("Starting Part 1: Re-ranking Products")

try:
    # Load data - WITH FILTER for testing
    df1 = spark.sql('''
        SELECT customerId, item.key as ranking, item.value as productCode 
        FROM database.schema.user_product_rankings
        LATERAL VIEW explode(map_entries(product_rankings)) as item
        WHERE customerId IN (
            SELECT DISTINCT user_id 
            FROM database.schema.add_to_cart_data
        )
    ''')
    
    df_3 = spark.sql('''
        SELECT customerId, norm_offline_transactions, norm_offline_atv, 
               norm_online_transactions, norm_online_atv 
        FROM database.schema.user_features
        WHERE customerId IN (
            SELECT DISTINCT user_id 
            FROM database.schema.add_to_cart_data
        )
    ''')
    
    df_4 = spark.sql('''
        SELECT productCode, norm_global_popularity, norm_co_occurrence, A_I, b_i 
        FROM database.schema.product_matrices
    ''')
    
    logger.info(f"Loaded data - df1: {df1.count()} rows, df_3: {df_3.count()} rows, df_4: {df_4.count()} rows")
    
    # Join all data at once
    df_all = df1.join(df_4, on='productCode', how='left') \
                .join(df_3, on='customerId', how='left')
    
    logger.info(f"Joined data - df_all: {df_all.count()} rows")
    
    # Create feature vector
    df_all = df_all.withColumn(
        'features',
        array(
            'norm_offline_transactions',
            'norm_offline_atv',
            'norm_online_transactions',
            'norm_online_atv',
            'norm_global_popularity',
            'norm_co_occurrence'
        )
    )
    
    # Calculate UCB scores using UDFs
    logger.info("Calculating theta values...")
    df_all = df_all.withColumn('theta', calculate_theta(col('A_I'), col('b_i')))
    
    logger.info("Calculating raw scores...")
    df_all = df_all.withColumn('raw_score', calculate_raw_score(col('theta'), col('features')))
    
    logger.info("Calculating uncertainty...")
    df_all = df_all.withColumn('uncertainty', calculate_uncertainty(col('A_I'), col('features')))
    
    logger.info("Calculating UCB scores...")
    df_all = df_all.withColumn('ucb_score', col('raw_score') + col('uncertainty'))
    
    # Calculate new rankings per customer
    window_spec = Window.partitionBy('customerId').orderBy(col('ucb_score').desc())
    df_all = df_all.withColumn('new_ranking', row_number().over(window_spec))
    
    # Select only needed columns
    df_ranking_updates = df_all.select('customerId', 'productCode', 'new_ranking')
    
    logger.info(f"Calculated rankings for {df_ranking_updates.count()} customer-product pairs")
    
    # Aggregate into map structure: {ranking -> productCode}
    df_ranking_map = df_ranking_updates.groupBy('customerId').agg(
        map_from_arrays(
            collect_list('new_ranking'),
            collect_list('productCode')
        ).alias('new_product_rankings')
    )
    
    # Create temporary view for merge
    df_ranking_map.createOrReplaceTempView('ranking_updates_temp')
    
    logger.info(f"Calculated new rankings for {df_ranking_map.count()} customers")
    
    # Batch update using Delta MERGE - Replace entire map
    spark.sql('''
        MERGE INTO database.schema.user_product_rankings AS target
        USING ranking_updates_temp AS source
        ON target.customerId = source.customerId
        WHEN MATCHED THEN UPDATE SET 
            target.product_rankings = source.new_product_rankings
        WHEN NOT MATCHED THEN INSERT (customerId, product_rankings)
            VALUES (source.customerId, source.new_product_rankings)
    ''')
    
    logger.info("Part 1 Complete: Re-ranking done successfully")
    
except Exception as e:
    logger.error(f"Error in Part 1: {str(e)}", exc_info=True)
    raise

# ============================================================================
# PART 2: LEARNING FROM CLICKS (UPDATE PHASE) - OPTIMIZED
# ============================================================================

logger.info("Starting Part 2: Learning from Clicks")

try:
    # Reload data
    df1 = spark.sql('''
        SELECT customerId, item.key as ranking, item.value as productCode 
        FROM database.schema.user_product_rankings
        LATERAL VIEW explode(map_entries(product_rankings)) as item
        WHERE customerId IN (
            SELECT DISTINCT user_id 
            FROM database.schema.add_to_cart_data
        )
    ''')
    
    # Load click data WITHOUT event_timestamp (doesn't exist)
    # Using monotonically_increasing_id() to create sequence for multiple clicks by same user
    df2 = spark.sql(''' 
        SELECT user_id as customerId, item_id as productCode
        FROM database.schema.add_to_cart_data
    ''')
    
    # Add click sequence using monotonically_increasing_id
    df2 = df2.withColumn('click_sequence', 
                         row_number().over(Window.partitionBy('customerId').orderBy(monotonically_increasing_id())))
    
    df_3 = spark.sql('''
        SELECT customerId, norm_offline_transactions, norm_offline_atv, 
               norm_online_transactions, norm_online_atv 
        FROM database.schema.user_features
        WHERE customerId IN (
            SELECT DISTINCT user_id 
            FROM database.schema.add_to_cart_data
        )
    ''')
    
    df_4 = spark.sql('''
        SELECT productCode, norm_global_popularity, norm_co_occurrence, A_I, b_i 
        FROM database.schema.product_matrices
    ''')
    
    logger.info(f"Loaded click data: {df2.count()} click events")
    
    # Join clicks with rankings to get the position
    df_clicks_with_position = df2.join(
        df1.withColumnRenamed('ranking', 'click_position'),
        on=['customerId', 'productCode'],
        how='inner'
    )
    
    logger.info(f"Matched {df_clicks_with_position.count()} clicks with rankings")
    
    # For each click, get all products shown before (cascade)
    # This is done by joining with all products where ranking <= click_position
    df_cascade = df_clicks_with_position.alias('clicks').join(
        df1.alias('all_products'),
        (col('clicks.customerId') == col('all_products.customerId')) &
        (col('all_products.ranking') <= col('clicks.click_position')),
        how='inner'
    ).select(
        col('clicks.customerId').alias('customerId'),
        col('clicks.productCode').alias('clicked_product'),
        col('clicks.click_position').alias('click_position'),
        col('clicks.click_sequence').alias('click_sequence'),
        col('all_products.productCode').alias('productCode'),
        col('all_products.ranking').alias('ranking')
    )
    
    logger.info(f"Created cascade view with {df_cascade.count()} product-click pairs (before reward assignment)")
    
    # Assign rewards: 1 for clicked product, 0 for others in cascade
    df_cascade = df_cascade.withColumn(
        'reward',
        when(col('productCode') == col('clicked_product'), 1).otherwise(0)
    )
    
    # Join with user features
    df_cascade = df_cascade.join(df_3, on='customerId', how='left')
    
    # Join with product features
    df_cascade = df_cascade.join(df_4, on='productCode', how='left')
    
    # Create feature vector
    df_cascade = df_cascade.withColumn(
        'features',
        array(
            'norm_offline_transactions',
            'norm_offline_atv',
            'norm_online_transactions',
            'norm_online_atv',
            'norm_global_popularity',
            'norm_co_occurrence'
        )
    )
    
    logger.info(f"Created cascade view with features: {df_cascade.count()} product-click pairs")
    
    # Update A_I and b_i using UDFs
    logger.info("Updating A_I matrices...")
    df_updates = df_cascade.withColumn('A_I_new', update_A_I(col('A_I'), col('features')))
    
    logger.info("Updating b_i vectors...")
    df_updates = df_updates.withColumn('b_i_new', update_b_i(col('b_i'), col('features'), col('reward')))
    
    # Aggregate updates per product (sum all updates for each product)
    logger.info("Aggregating updates per product...")
    df_aggregated_updates = df_updates.groupBy('productCode').agg(
        first('A_I').alias('A_I_original'),
        first('b_i').alias('b_i_original'),
        collect_list('A_I_new').alias('A_I_updates'),
        collect_list('b_i_new').alias('b_i_updates')
    )
    
    # UDF to aggregate all updates for a product - Returns 2D array
    @udf(ArrayType(ArrayType(DecimalType(10, 6))))
    def aggregate_matrix_updates(original, updates):
        try:
            # Start with original matrix (2D)
            A_I_flat = [item for sublist in original for item in sublist]
            result = np.array(A_I_flat, dtype=float).reshape(6, 6)
            
            # Apply each update incrementally
            for update in updates:
                # Each update is 2D array
                update_flat = [item for sublist in update for item in sublist]
                update_array = np.array(update_flat, dtype=float).reshape(6, 6)
                result = update_array  # Take the last update which has all accumulated changes
            
            # Convert to 2D list with Decimal type
            return [[Decimal(str(round(float(val), 6))) for val in row] for row in result]
        except Exception as e:
            logger.error(f"Error aggregating matrix updates: {str(e)}")
            return original
    
    # UDF to aggregate vector updates - Returns 1D array
    @udf(ArrayType(DecimalType(10, 6)))
    def aggregate_vector_updates(original, updates):
        try:
            # Start with original vector
            result = np.array(original, dtype=float)
            
            # Apply each update incrementally
            for update in updates:
                result = np.array(update, dtype=float)  # Take the last update
            
            # Convert to list with Decimal type
            return [Decimal(str(round(float(val), 6))) for val in result]
        except Exception as e:
            logger.error(f"Error aggregating vector updates: {str(e)}")
            return original
    
    logger.info("Finalizing aggregated updates...")
    df_aggregated_updates = df_aggregated_updates.withColumn(
        'A_I_final',
        aggregate_matrix_updates(col('A_I_original'), col('A_I_updates'))
    )
    df_aggregated_updates = df_aggregated_updates.withColumn(
        'b_i_final',
        aggregate_vector_updates(col('b_i_original'), col('b_i_updates'))
    )
    
    # Prepare final updates
    df_final_updates = df_aggregated_updates.select(
        'productCode',
        col('A_I_final').alias('A_I'),
        col('b_i_final').alias('b_i')
    )
    
    # Create temporary view for merge
    df_final_updates.createOrReplaceTempView('weight_updates_temp')
    
    logger.info(f"Prepared weight updates for {df_final_updates.count()} products")
    
    # Batch update using Delta MERGE
    spark.sql('''
        MERGE INTO database.schema.product_matrices AS target
        USING weight_updates_temp AS source
        ON target.productCode = source.productCode
        WHEN MATCHED THEN UPDATE SET 
            target.A_I = source.A_I,
            target.b_i = source.b_i
    ''')
    
    logger.info("Part 2 Complete: Weight updates done successfully")
    
    # Log statistics
    update_stats = df_cascade.groupBy('reward').count().collect()
    for stat in update_stats:
        logger.info(f"Reward {stat['reward']}: {stat['count']} updates")
    
except Exception as e:
    logger.error(f"Error in Part 2: {str(e)}", exc_info=True)
    raise

# ============================================================================
# FINAL VALIDATION AND LOGGING
# ============================================================================

try:
    # Validate updates
    validation_query = spark.sql('''
        SELECT 
            COUNT(DISTINCT customerId) as total_customers,
            AVG(size(product_rankings)) as avg_products_per_customer
        FROM database.schema.user_product_rankings
    ''')
    
    validation_results = validation_query.collect()[0]
    
    logger.info(f"Validation - Total Customers: {validation_results['total_customers']}, "
                f"Avg Products per Customer: {validation_results['avg_products_per_customer']}")
    
    logger.info("=" * 80)
    logger.info("CASCADING BANDITS UPDATE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    
except Exception as e:
    logger.error(f"Error in validation: {str(e)}", exc_info=True)
