# tools/neo4j_setup.py

import os
import logging
from typing import Optional
from functools import lru_cache

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

# Load environment variables
load_dotenv()

##############################################################################
# 1) Setup: Neo4j, Logging, and Environment
##############################################################################

NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Validate environment variables
missing_env_vars = []
if not NEO4J_URI:
    missing_env_vars.append("NEO4J_URI")
if not NEO4J_USERNAME:
    missing_env_vars.append("NEO4J_USERNAME")
if not NEO4J_PASSWORD:
    missing_env_vars.append("NEO4J_PASSWORD")
if not OPENAI_API_KEY:
    missing_env_vars.append("OPENAI_API_KEY")

if missing_env_vars:
    logging.warning(f"⚠️  Missing environment variables: {missing_env_vars}")
    logging.warning("   Some features may not work until these are set.")

logger = logging.getLogger(__name__)

# Global variables for lazy initialization
_graph: Optional[Neo4jGraph] = None
_neo4j_vector: Optional[Neo4jVector] = None
_connection_validated: bool = False


def validate_connection(uri: str = None, username: str = None, password: str = None) -> bool:
    """
    Validate Neo4j connection with proper error handling.
    
    Args:
        uri: Neo4j URI (defaults to NEO4J_URI)
        username: Username (defaults to NEO4J_USERNAME)
        password: Password (defaults to NEO4J_PASSWORD)
    
    Returns:
        True if connection successful, False otherwise
    """
    uri = uri or NEO4J_URI
    username = username or NEO4J_USERNAME
    password = password or NEO4J_PASSWORD
    
    if not password:
        logger.error("❌ NEO4J_PASSWORD is not set")
        return False
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session(database=NEO4J_DATABASE) as session:
            # Simple query to test connection
            session.run("RETURN 1 AS test")
        driver.close()
        logger.info("✅ Neo4j connection validated successfully")
        return True
    except AuthError as e:
        logger.error(f"❌ Neo4j authentication failed: {e}")
        logger.error("   Please check your NEO4J_USERNAME and NEO4J_PASSWORD in .env file")
        return False
    except ServiceUnavailable as e:
        logger.error(f"❌ Neo4j service unavailable: {e}")
        logger.error("   Please check your NEO4J_URI and ensure the database is running")
        return False
    except Exception as e:
        logger.error(f"❌ Neo4j connection error: {e}")
        return False


def create_constraints(uri: str = None, user: str = None, pwd: str = None) -> bool:
    """
    Create unique constraints or indexes as needed for production usage.
    Uses lazy initialization - only connects when needed.
    
    Args:
        uri: Neo4j URI (defaults to NEO4J_URI)
        user: Username (defaults to NEO4J_USERNAME)
        pwd: Password (defaults to NEO4J_PASSWORD)
    
    Returns:
        True if successful, False otherwise
    """
    uri = uri or NEO4J_URI
    user = user or NEO4J_USERNAME
    pwd = pwd or NEO4J_PASSWORD
    
    if not pwd:
        logger.error("❌ Cannot create constraints: NEO4J_PASSWORD not set")
        return False
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        with driver.session(database=NEO4J_DATABASE) as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (co:Country) REQUIRE co.name IS UNIQUE")
        driver.close()
        logger.info("✅ Neo4j constraints created/ensured.")
        return True
    except AuthError as e:
        logger.error(f"❌ Failed to create constraints: Authentication error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to create constraints: {e}")
        return False


def _initialize_graph() -> Optional[Neo4jGraph]:
    """
    Internal function to initialize Neo4jGraph.
    Only connects when first accessed.
    
    Returns:
        Neo4jGraph instance or None if connection fails
    """
    global _graph
    
    if _graph is not None:
        return _graph
    
    if not NEO4J_PASSWORD:
        logger.error("❌ Cannot create Neo4jGraph: NEO4J_PASSWORD not set")
        return None
    
    # Validate connection first
    if not validate_connection():
        logger.error("❌ Cannot create Neo4jGraph: Connection validation failed")
        return None
    
    try:
        # Create constraints before building graph
        create_constraints()
        
        _graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            refresh_schema=False
        )
        logger.info("✅ Neo4jGraph initialized successfully")
        return _graph
    except AuthError as e:
        logger.error(f"❌ Failed to create Neo4jGraph: Authentication error: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to create Neo4jGraph: {e}")
        return None


def _initialize_neo4j_vector() -> Optional[Neo4jVector]:
    """
    Internal function to initialize Neo4jVector.
    Only connects when first accessed.
    
    Returns:
        Neo4jVector instance or None if connection fails
    """
    global _neo4j_vector
    
    if _neo4j_vector is not None:
        return _neo4j_vector
    
    if not NEO4J_PASSWORD:
        logger.error("❌ Cannot create Neo4jVector: NEO4J_PASSWORD not set")
        return None
    
    if not OPENAI_API_KEY:
        logger.error("❌ Cannot create Neo4jVector: OPENAI_API_KEY not set")
        return None
    
    # Ensure graph is initialized first
    if _initialize_graph() is None:
        logger.error("❌ Cannot create Neo4jVector: Neo4jGraph not available")
        return None
    
    try:
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
        
        _neo4j_vector = Neo4jVector.from_existing_graph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            embedding=embedding_model,
            index_name="company_unstructured_idx",
            node_label="Company",
            text_node_properties=["unstructured_data"],
            embedding_node_property="embedding"
        )
        logger.info("✅ Neo4jVector initialized successfully")
        return _neo4j_vector
    except AuthError as e:
        logger.error(f"❌ Failed to create Neo4jVector: Authentication error: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to create Neo4jVector: {e}")
        return None


# Module-level accessors using __getattr__ for lazy initialization
# This allows `from tools.neo4j_setup import graph` to work lazily
def __getattr__(name: str):
    """
    Lazy attribute access for 'graph' and 'neo4j_vector'.
    This allows the module to be imported even if Neo4j is unavailable.
    """
    if name == "graph":
        return _initialize_graph()
    elif name == "neo4j_vector":
        return _initialize_neo4j_vector()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Initialize connection validation on import (non-blocking)
# This allows the module to load even if Neo4j is temporarily unavailable
try:
    if NEO4J_PASSWORD and OPENAI_API_KEY:
        # Only validate connection, don't create objects yet
        _connection_validated = validate_connection()
        if _connection_validated:
            logger.info("✅ Neo4j connection ready (lazy initialization enabled)")
        else:
            logger.warning("⚠️  Neo4j connection validation failed - will retry on first use")
    else:
        logger.warning("⚠️  Neo4j credentials not fully configured - will retry on first use")
except Exception as e:
    logger.warning(f"⚠️  Neo4j setup warning: {e} - will retry on first use")

##############################################################################
# Shutdown Function
##############################################################################

def shutdown():
    """
    Attempts to close persistent connections to avoid resource warnings.
    """
    global _graph, _neo4j_vector
    
    try:
        if _graph is not None and hasattr(_graph, "close") and callable(_graph.close):
            _graph.close()
            logger.info("Graph connection closed.")
        elif _graph is not None:
            logger.info("Graph object has no close() method.")
    except Exception as e:
        logger.error(f"Error closing graph: {e}")
    
    try:
        if _neo4j_vector is not None and hasattr(_neo4j_vector, "close") and callable(_neo4j_vector.close):
            _neo4j_vector.close()
            logger.info("Neo4j vector connection closed.")
        elif _neo4j_vector is not None:
            logger.info("Neo4j vector object has no close() method.")
    except Exception as e:
        logger.error(f"Error closing neo4j_vector: {e}")
    
    # Clear cached instances
    _graph = None
    _neo4j_vector = None
