# agents/sourcing_agent.py

import logging
from typing import Dict, Any, List
from crewai import Agent

# Configure logging
logging.basicConfig(level=logging.INFO)

class SourcingAgent(Agent):
    """
    Sourcing Agent:
    Executes decisions from the ChiefSupplyChainAgent by finding and integrating
    alternative suppliers for high-risk suppliers.
    """
    def __init__(self, **config):
        """
        Initialize the SourcingAgent with configuration.
        
        Args:
            **config: Configuration parameters from agents.yaml (e.g., role, goal, backstory).
        """
        super().__init__(**config)
        logging.info("🚀 SourcingAgent initialized")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task: source alternative suppliers based on decisions.
        
        Args:
            inputs: Dictionary containing the 'decisions' from the ChiefSupplyChainAgent.
        
        Returns:
            Dictionary with 'sourcing_results' key mapping suppliers to new suppliers or errors.
        """
        decisions = inputs.get("decisions", {})
        if not decisions:
            logging.error("No decisions provided for sourcing actions.")
            return {"error": "No decisions provided."}

        sourcing_results = {}
        for supplier, decision in decisions.items():
            if decision.get("action") == "replace_supplier":
                alternatives = self._search_alternative_suppliers(supplier)
                best_supplier = self._evaluate_suppliers(alternatives)
                if best_supplier:
                    sourcing_results[supplier] = {"new_supplier": best_supplier}
                    # Placeholder for integration logic (e.g., update Neo4j or supply chain map)
                    logging.info(f"Integrated {best_supplier} as alternative for {supplier}.")
                else:
                    sourcing_results[supplier] = {"error": "No suitable alternative found."}
                    logging.warning(f"No suitable alternative found for {supplier}.")
            else:
                # Skip suppliers not requiring replacement
                continue
        return {"sourcing_results": sourcing_results}

    def _search_alternative_suppliers(self, supplier: str) -> List[Dict[str, Any]]:
        """
        Mock function to search for alternative suppliers.
        In a real system, this would query a supplier database or API.
        
        Args:
            supplier: Name of the supplier to replace.
        
        Returns:
            List of dictionaries, each representing a potential supplier with attributes.
        """
        # Mock data (replace with real API call in production)
        return [
            {"name": "AltSupplier1", "country": "Germany", "cost": 100, "lead_time": 5, "stability": 0.9},
            {"name": "AltSupplier2", "country": "USA", "cost": 120, "lead_time": 7, "stability": 0.85},
            {"name": "AltSupplier3", "country": "China", "cost": 80, "lead_time": 10, "stability": 0.7}
        ]

    def _evaluate_suppliers(self, alternatives: List[Dict[str, Any]]) -> str:
        """
        Evaluate alternative suppliers and select the best one based on stability.
        
        Args:
            alternatives: List of potential suppliers with attributes.
        
        Returns:
            Name of the selected supplier, or None if no suitable supplier is found.
        """
        if not alternatives:
            return None
        # Select supplier with highest stability (can be extended with multi-criteria logic)
        best_supplier = max(alternatives, key=lambda x: x["stability"])
        return best_supplier["name"]

# Example usage (for testing purposes)
if __name__ == "__main__":
    agent = SourcingAgent(
        role="Sourcing Specialist",
        goal="Source reliable alternative suppliers",
        backstory="Expert in global supplier networks and evaluation"
    )
    sample_input = {
        "decisions": {
            "SupplierA": {"action": "replace_supplier", "priority": "high"},
            "SupplierB": {"action": "increase_inventory", "priority": "medium"}
        }
    }
    result = agent.execute(sample_input)
    print(result)