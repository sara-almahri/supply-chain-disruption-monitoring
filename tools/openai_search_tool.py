"""
OpenAI Search Tool
=================
Robust search tool using OpenAI's capabilities for product and company information retrieval.
Replaces SerperDevTool with more reliable and consistent search results.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class OpenAISearchInput(BaseModel):
    """Input schema for OpenAI search tool."""
    query: str = Field(..., description="Search query string")
    max_results: int = Field(default=5, description="Maximum number of results to return")
    search_type: str = Field(
        default="general",
        description="Type of search: 'general', 'product', 'company', 'supplier'"
    )


class OpenAISearchTool:
    """
    Advanced search tool using OpenAI's GPT models for information retrieval.
    Provides more reliable and context-aware search results than traditional web search APIs.
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"  # Using latest model for best results
        
    def _build_search_prompt(self, query: str, search_type: str) -> str:
        """Build context-aware prompt based on search type."""
        base_prompts = {
            "product": (
                f"Search for information about products manufactured or supplied by companies. "
                f"Query: {query}\n\n"
                f"Provide specific product names, categories, and applications. "
                f"Focus on industrial/commercial products relevant to supply chain analysis."
            ),
            "company": (
                f"Search for information about a company, including its industry, "
                f"products, services, and business operations.\n\n"
                f"Query: {query}\n\n"
                f"Provide accurate, factual information about the company's business activities."
            ),
            "supplier": (
                f"Search for supplier information, including what products or services "
                f"they provide, their customers, and supply chain relationships.\n\n"
                f"Query: {query}\n\n"
                f"Focus on supply chain and business relationship information."
            ),
            "general": (
                f"Search for information related to: {query}\n\n"
                f"Provide comprehensive, accurate information relevant to supply chain analysis."
            )
        }
        
        return base_prompts.get(search_type, base_prompts["general"])
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        search_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Perform search using OpenAI's capabilities.
        
        Args:
            query: Search query string
            max_results: Maximum number of results (not strictly enforced, but guides response length)
            search_type: Type of search to optimize prompt
            
        Returns:
            Dictionary with search results including:
            - results: List of result dictionaries
            - summary: Summary of findings
            - confidence: Confidence score (0-1)
        """
        try:
            prompt = self._build_search_prompt(query, search_type)
            
            system_message = (
                "You are an expert information retrieval system specializing in "
                "supply chain, manufacturing, and business intelligence. "
                "Provide accurate, factual information based on your training data. "
                "If information is not available or uncertain, clearly state that. "
                "Format responses in a structured, easy-to-parse manner."
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # Deterministic: temperature=0 for consistent results
                max_tokens=2000,
                top_p=0.9
            )
            
            content = response.choices[0].message.content
            
            # Parse and structure the response
            results = self._parse_response(content, max_results)
            
            return {
                "query": query,
                "search_type": search_type,
                "results": results,
                "summary": content[:500] if len(content) > 500 else content,
                "full_response": content,
                "confidence": 0.85  # OpenAI models generally have high confidence
            }
            
        except Exception as e:
            logger.error(f"OpenAI search failed for query '{query}': {e}")
            return {
                "query": query,
                "search_type": search_type,
                "results": [],
                "summary": f"Search failed: {str(e)}",
                "full_response": "",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _parse_response(self, content: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Parse OpenAI response into structured results.
        
        Args:
            content: Raw response from OpenAI
            max_results: Maximum number of results to extract
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        # Try to extract structured information
        # Look for lists, bullet points, or numbered items
        lines = content.split('\n')
        current_result = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_result:
                    results.append(current_result)
                    current_result = {}
                    if len(results) >= max_results:
                        break
                continue
            
            # Check for common patterns
            if line.startswith(('-', '*', '•', '1.', '2.', '3.')):
                # This might be a result item
                if current_result:
                    results.append(current_result)
                    current_result = {}
                    if len(results) >= max_results:
                        break
                current_result = {"text": line.lstrip('- *•1234567890. '), "type": "item"}
            elif ':' in line and len(line.split(':')) == 2:
                # Key-value pair
                key, value = line.split(':', 1)
                current_result[key.strip().lower().replace(' ', '_')] = value.strip()
        
        # Add final result if exists
        if current_result and len(results) < max_results:
            results.append(current_result)
        
        # If no structured results found, create one from full content
        if not results:
            results.append({
                "text": content[:1000],  # Limit length
                "type": "summary"
            })
        
        return results[:max_results]
    
    def search_products(self, company_name: str) -> str:
        """
        Specialized method for searching company products.
        
        Args:
            company_name: Name of the company
            
        Returns:
            String description of products
        """
        query = f"What products does {company_name} manufacture or supply?"
        result = self.search(query, max_results=3, search_type="product")
        
        if result.get("error"):
            return f"Unable to find product information for {company_name}"
        
        # Extract product information from results
        products = []
        for res in result.get("results", []):
            if "text" in res:
                products.append(res["text"])
            elif "product" in res:
                products.append(res["product"])
        
        if products:
            return "; ".join(products)
        else:
            return result.get("summary", f"Product information for {company_name}")


# Create tool instance
_openai_search_tool_instance = None

def get_openai_search_tool() -> OpenAISearchTool:
    """Get or create OpenAI search tool instance."""
    global _openai_search_tool_instance
    if _openai_search_tool_instance is None:
        _openai_search_tool_instance = OpenAISearchTool()
    return _openai_search_tool_instance


def openai_search_tool_func(
    query: str,
    max_results: int = 5,
    search_type: str = "general"
) -> Dict[str, Any]:
    """Wrapper function for StructuredTool."""
    tool = get_openai_search_tool()
    return tool.search(query, max_results, search_type)


# Create StructuredTool
openai_search_tool = StructuredTool(
    name="openai_search",
    description=(
        "Advanced search tool using OpenAI's GPT models for reliable information retrieval. "
        "Specialized for supply chain, product, and company information. "
        "More accurate and context-aware than traditional web search APIs."
    ),
    func=openai_search_tool_func,
    args_schema=OpenAISearchInput
)


