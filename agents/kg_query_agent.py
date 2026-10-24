import logging

from crewai import Agent



class KGQueryAgent(Agent):
    """
    Supply chain pathfinder that uses build_disrupted_supply_chains tool
    to retrieve complete disrupted supply chain paths (Tier-1 to Tier-4).
    """

    def __init__(self, **config):
        super().__init__(**config)
        logging.info("🚀 KGQueryAgent (Tier-1..4 comprehensive pathfinder) initialized")
