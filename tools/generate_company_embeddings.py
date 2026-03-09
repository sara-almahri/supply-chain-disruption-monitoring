"""
tools/generate_company_embeddings.py
------------------------------------

One-time script to generate and store OpenAI embeddings for all companies in Neo4j.
This enables semantic similarity-based entity resolution.

Usage:
    python tools/generate_company_embeddings.py
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI
from tqdm import tqdm

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from tools.neo4j_setup import graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_all_companies() -> List[Dict[str, str]]:
    """Fetch all companies from Neo4j."""
    query = """
    MATCH (c:Company)
    OPTIONAL MATCH (c)-[:locatedIn]->(country:Country)
    RETURN c.name AS name, 
           c.industry AS industry,
           collect(DISTINCT country.name) AS countries
    ORDER BY c.name
    """
    try:
        results = graph.query(query)
        logger.info(f"✅ Retrieved {len(results)} companies from Neo4j")
        return results
    except Exception as e:
        logger.error(f"❌ Failed to fetch companies: {e}")
        return []


def create_company_text_representation(company: Dict[str, Any]) -> str:
    """
    Create a rich text representation of the company for embedding.
    This helps the model understand context beyond just the name.
    """
    name = company.get("name", "")
    industry = company.get("industry", "")
    countries = company.get("countries", [])
    
    # Rich representation for better semantic matching
    text_parts = [f"Company: {name}"]
    
    if industry:
        text_parts.append(f"Industry: {industry}")
    
    if countries:
        countries_str = ", ".join([c for c in countries if c])
        if countries_str:
            text_parts.append(f"Located in: {countries_str}")
    
    return " | ".join(text_parts)


def generate_embeddings_batch(
    texts: List[str],
    model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts using OpenAI API.
    
    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model to use
        
    Returns:
        List of embedding vectors
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    try:
        response = client.embeddings.create(
            input=texts,
            model=model
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"❌ Failed to generate embeddings: {e}")
        raise


def store_embeddings_in_neo4j(
    company_embeddings: List[Dict[str, Any]],
    batch_size: int = 100
) -> None:
    """
    Store embeddings in Neo4j as node properties.
    
    Args:
        company_embeddings: List of dicts with 'name', 'text', and 'embedding'
        batch_size: Number of nodes to update per transaction
    """
    query = """
    UNWIND $batch AS item
    MATCH (c:Company {name: item.name})
    SET c.embedding_text = item.text,
        c.embedding = item.embedding,
        c.embedding_model = item.model,
        c.embedding_updated = datetime()
    RETURN count(c) as updated
    """
    
    total = len(company_embeddings)
    for i in range(0, total, batch_size):
        batch = company_embeddings[i:i + batch_size]
        try:
            result = graph.query(query, {"batch": batch})
            updated = result[0].get("updated", 0) if result else 0
            logger.info(f"✅ Stored embeddings for batch {i//batch_size + 1} ({updated} companies)")
        except Exception as e:
            logger.error(f"❌ Failed to store batch {i//batch_size + 1}: {e}")
            raise


def create_vector_index() -> None:
    """
    Create a vector index in Neo4j for fast similarity search.
    Requires Neo4j 5.11+ with vector index support.
    """
    # First, check if index exists
    check_query = """
    SHOW INDEXES
    YIELD name, type
    WHERE name = 'company_embedding_index'
    RETURN count(*) as exists
    """
    
    try:
        result = graph.query(check_query)
        exists = result[0].get("exists", 0) if result else 0
        
        if exists > 0:
            logger.info("✅ Vector index already exists")
            return
        
        # Create vector index (Neo4j 5.11+)
        create_index_query = """
        CREATE VECTOR INDEX company_embedding_index IF NOT EXISTS
        FOR (c:Company)
        ON c.embedding
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
        """
        
        graph.query(create_index_query)
        logger.info("✅ Created vector index for company embeddings")
        
        # Wait for index to be online
        logger.info("⏳ Waiting for index to come online...")
        time.sleep(5)
        
    except Exception as e:
        logger.warning(f"⚠️  Could not create vector index (requires Neo4j 5.11+): {e}")
        logger.info("💡 Similarity search will still work but may be slower")


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("COMPANY EMBEDDING GENERATOR")
    logger.info("=" * 80)
    logger.info("")
    
    # Check OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY environment variable not set")
        return
    
    # Step 1: Fetch all companies
    logger.info("📊 Step 1: Fetching companies from Neo4j...")
    companies = get_all_companies()
    
    if not companies:
        logger.error("❌ No companies found")
        return
    
    logger.info(f"✅ Found {len(companies)} companies")
    logger.info("")
    
    # Step 2: Generate embeddings
    logger.info("🧠 Step 2: Generating embeddings...")
    logger.info(f"   Model: text-embedding-3-small")
    logger.info(f"   Estimated cost: ${len(companies) * 0.00002:.4f}")
    logger.info("")
    
    # Batch processing for efficiency
    batch_size = 100
    company_embeddings = []
    
    for i in tqdm(range(0, len(companies), batch_size), desc="Generating embeddings"):
        batch = companies[i:i + batch_size]
        
        # Create text representations
        texts = [create_company_text_representation(c) for c in batch]
        
        # Generate embeddings
        try:
            embeddings = generate_embeddings_batch(texts)
            
            # Store results
            for company, text, embedding in zip(batch, texts, embeddings):
                company_embeddings.append({
                    "name": company["name"],
                    "text": text,
                    "embedding": embedding,
                    "model": "text-embedding-3-small"
                })
            
            # Rate limiting (avoid hitting OpenAI limits)
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"❌ Failed to process batch {i//batch_size + 1}: {e}")
            continue
    
    logger.info(f"✅ Generated {len(company_embeddings)} embeddings")
    logger.info("")
    
    # Step 3: Store in Neo4j
    logger.info("💾 Step 3: Storing embeddings in Neo4j...")
    store_embeddings_in_neo4j(company_embeddings)
    logger.info("✅ All embeddings stored successfully")
    logger.info("")
    
    # Step 4: Create vector index
    logger.info("🔍 Step 4: Creating vector index...")
    create_vector_index()
    logger.info("")
    
    # Step 5: Verification
    logger.info("✔️  Step 5: Verification...")
    verify_query = """
    MATCH (c:Company)
    WHERE c.embedding IS NOT NULL
    RETURN count(c) as companies_with_embeddings
    """
    result = graph.query(verify_query)
    count = result[0].get("companies_with_embeddings", 0) if result else 0
    
    logger.info(f"✅ {count}/{len(companies)} companies now have embeddings")
    logger.info("")
    
    logger.info("=" * 80)
    logger.info("✅ EMBEDDING GENERATION COMPLETE!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. The entity resolution function will now use semantic similarity")
    logger.info("  2. Test with: python -c 'from tools.full_supply_chain_path_tool import build_disrupted_supply_chains; ...'")
    logger.info("")


if __name__ == "__main__":
    main()

