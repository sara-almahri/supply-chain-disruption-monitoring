"""
Knowledge Graph Data Ingestion Script
=====================================
Robust, production-ready script for ingesting supply chain CSV data into Neo4j.
Includes comprehensive error handling, validation, and progress tracking.
"""

import csv
import os
import logging
from typing import Dict, List, Tuple
from neo4j import GraphDatabase, exceptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Neo4j Connection Details
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Dataset path
CSV_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dataset",
    "supplychainKG.csv"
)


def validate_connection(uri: str, username: str, password: str) -> bool:
    """Validate Neo4j connection before ingestion."""
    if not password:
        logger.error("❌ NEO4J_PASSWORD is empty! Please set it in your .env file.")
        return False
    
    try:
        logger.info(f"Attempting to connect to Neo4j at {uri}...")
        logger.info(f"Username: {username}")
        logger.debug(f"Password length: {len(password)} characters")
        
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            result.single()
        driver.close()
        logger.info("✅ Neo4j connection validated successfully")
        return True
    except exceptions.AuthError as e:
        logger.error(f"❌ Neo4j authentication failed!")
        logger.error(f"   Error: {e}")
        logger.error(f"   Please verify your NEO4J_USERNAME and NEO4J_PASSWORD in the .env file")
        logger.error(f"   URI: {uri}")
        logger.error(f"   Username: {username}")
        return False
    except Exception as e:
        logger.error(f"❌ Neo4j connection failed: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        return False


def create_constraints_and_indexes(driver: GraphDatabase.driver) -> None:
    """Create unique constraints and indexes for optimal performance."""
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (co:Country) REQUIRE co.name IS UNIQUE",
    ]
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS FOR (c:Company) ON (c.industry)",
        "CREATE INDEX IF NOT EXISTS FOR (c:Company) ON (c.name)",
        "CREATE INDEX IF NOT EXISTS FOR (co:Country) ON (co.name)",
    ]
    
    with driver.session() as session:
        for constraint in constraints:
            try:
                session.run(constraint)
                logger.debug(f"Created constraint/index: {constraint[:50]}...")
            except Exception as e:
                logger.warning(f"Constraint/index may already exist: {e}")


def validate_csv_row(row: Dict[str, str]) -> Tuple[bool, str]:
    """Validate a CSV row has all required fields."""
    required_fields = [
        "Supplier", "Supplier Industry", "Supplier Country",
        "Customer", "Customer Industry", "Customer Country"
    ]
    
    for field in required_fields:
        if not row.get(field) or not row[field].strip():
            return False, f"Missing or empty field: {field}"
    
    return True, ""


def ingest_supply_chain_data(
    uri: str,
    username: str,
    password: str,
    csv_path: str,
    batch_size: int = 1000
) -> Dict[str, int]:
    """
    Robust CSV ingestion with batch processing and error handling.
    
    Args:
        uri: Neo4j connection URI
        username: Neo4j username
        password: Neo4j password
        csv_path: Path to CSV file
        batch_size: Number of rows to process in each batch
        
    Returns:
        Dictionary with statistics: total_rows, successful, failed, skipped
    """
    stats = {
        "total_rows": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "companies_created": 0,
        "countries_created": 0,
        "relationships_created": 0
    }
    
    if not os.path.exists(csv_path):
        logger.error(f"❌ CSV file not found: {csv_path}")
        return stats
    
    # Validate connection
    if not validate_connection(uri, username, password):
        return stats
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    try:
        # Create constraints and indexes
        create_constraints_and_indexes(driver)
        
        # Batch processing query
        batch_query = """
        UNWIND $batch AS row
        MERGE (s:Company {name: row.supplier_name})
        ON CREATE SET 
            s.industry = row.supplier_industry,
            s.created_at = datetime()
        ON MATCH SET
            s.industry = COALESCE(s.industry, row.supplier_industry)
        
        MERGE (c:Company {name: row.customer_name})
        ON CREATE SET 
            c.industry = row.customer_industry,
            c.created_at = datetime()
        ON MATCH SET
            c.industry = COALESCE(c.industry, row.customer_industry)
        
        MERGE (sc:Country {name: row.supplier_country})
        ON CREATE SET sc.created_at = datetime()
        
        MERGE (cc:Country {name: row.customer_country})
        ON CREATE SET cc.created_at = datetime()
        
        MERGE (s)-[:suppliesTo]->(c)
        ON CREATE SET 
            s.supplier_count = COALESCE(s.supplier_count, 0) + 1,
            c.customer_count = COALESCE(c.customer_count, 0) + 1
        
        MERGE (s)-[:locatedIn]->(sc)
        MERGE (c)-[:locatedIn]->(cc)
        """
        
        batch = []
        
        with open(csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                stats["total_rows"] += 1
                
                # Validate row
                is_valid, error_msg = validate_csv_row(row)
                if not is_valid:
                    logger.warning(f"Row {row_num} skipped: {error_msg}")
                    stats["skipped"] += 1
                    continue
                
                # Prepare batch entry
                batch_entry = {
                    "supplier_name": row["Supplier"].strip(),
                    "supplier_industry": row["Supplier Industry"].strip(),
                    "supplier_country": row["Supplier Country"].strip(),
                    "customer_name": row["Customer"].strip(),
                    "customer_industry": row["Customer Industry"].strip(),
                    "customer_country": row["Customer Country"].strip()
                }
                
                batch.append(batch_entry)
                
                # Process batch when it reaches batch_size
                if len(batch) >= batch_size:
                    try:
                        with driver.session() as session:
                            result = session.run(batch_query, {"batch": batch})
                            result.consume()  # Consume result to execute
                        stats["successful"] += len(batch)
                        logger.debug(f"Processed batch: {len(batch)} rows")
                    except exceptions.Neo4jError as e:
                        logger.error(f"Neo4j error in batch: {e}")
                        stats["failed"] += len(batch)
                    except Exception as e:
                        logger.error(f"Unexpected error in batch: {e}")
                        stats["failed"] += len(batch)
                    
                    batch = []
                    
                    # Progress update
                    if stats["total_rows"] % 5000 == 0:
                        logger.info(f"Progress: {stats['total_rows']} rows processed, "
                                  f"{stats['successful']} successful, "
                                  f"{stats['failed']} failed")
            
            # Process remaining batch
            if batch:
                try:
                    with driver.session() as session:
                        result = session.run(batch_query, {"batch": batch})
                        result.consume()
                    stats["successful"] += len(batch)
                except Exception as e:
                    logger.error(f"Error processing final batch: {e}")
                    stats["failed"] += len(batch)
        
        # Get final statistics
        with driver.session() as session:
            company_count = session.run("MATCH (c:Company) RETURN count(c) AS count").single()["count"]
            country_count = session.run("MATCH (co:Country) RETURN count(co) AS count").single()["count"]
            rel_count = session.run("MATCH ()-[r:suppliesTo]->() RETURN count(r) AS count").single()["count"]
            
            stats["companies_created"] = company_count
            stats["countries_created"] = country_count
            stats["relationships_created"] = rel_count
        
        logger.info("✅ Data ingestion complete!")
        logger.info(f"   Total rows processed: {stats['total_rows']}")
        logger.info(f"   Successful: {stats['successful']}")
        logger.info(f"   Failed: {stats['failed']}")
        logger.info(f"   Skipped: {stats['skipped']}")
        logger.info(f"   Companies in graph: {stats['companies_created']}")
        logger.info(f"   Countries in graph: {stats['countries_created']}")
        logger.info(f"   Relationships in graph: {stats['relationships_created']}")
        
    except Exception as e:
        logger.error(f"❌ Fatal error during ingestion: {e}", exc_info=True)
    finally:
        driver.close()
    
    return stats


if __name__ == "__main__":
    if not NEO4J_PASSWORD:
        logger.error("❌ NEO4J_PASSWORD environment variable not set")
        logger.error("   Please set NEO4J_PASSWORD in your .env file")
        logger.error("   Example: NEO4J_PASSWORD=your_password_here")
        exit(1)
    
    logger.info("🚀 Starting Knowledge Graph ingestion...")
    logger.info(f"   CSV file: {CSV_FILE}")
    logger.info(f"   Neo4j URI: {NEO4J_URI}")
    logger.info(f"   Username: {NEO4J_USERNAME}")
    
    # Validate connection first
    if not validate_connection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD):
        logger.error("❌ Cannot proceed without valid Neo4j connection")
        logger.error("   Please check your credentials and try again")
        exit(1)
    
    stats = ingest_supply_chain_data(
        NEO4J_URI,
        NEO4J_USERNAME,
        NEO4J_PASSWORD,
        CSV_FILE,
        batch_size=1000
    )
    
    if stats["failed"] > 0:
        logger.warning(f"⚠️  {stats['failed']} rows failed to ingest")
        exit(1)
    else:
        logger.info("✅ All data ingested successfully!")
        exit(0)

