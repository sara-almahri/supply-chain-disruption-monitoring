# crew.py

import yaml
from crewai import Agent, Task, Crew
from crewai.project import CrewBase, agent, crew, task
from crewai import Process

# Tools
from tools.disruption_analysis_tool import DisruptionAnalysisTool
from tools.text_processor_tool import TextProcessorTool
from tools.crewai_tool_wrapper import wrap_tool
from tools.entity_resolver_tool import resolve_entity_struct
from tools.tier_calculator_tool import calculate_tier_struct
from tools.disruption_impact_tool import disruption_impact_tool
from tools.graph_metrics_tool import graph_metrics_tool
from tools.enhanced_graph_metrics_tool import enhanced_graph_metrics_tool
from tools.openai_search_tool import openai_search_tool

# Custom Agents
from agents.chief_supply_chain_agent import ChiefSupplyChainAgent

def load_yaml_config(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def load_company_config():
    """Load company configuration."""
    try:
        config = load_yaml_config("config/company_config.yaml")
        return config.get("company", {})
    except Exception as e:
        return {"name": "Tesla", "settings": {"max_tier_depth": 4}}

@CrewBase
class SupplyChainCrew:
    def __init__(self, company_name: str = None):
        company_config = load_company_config()
        self.company_name = company_name or company_config.get("name", "Tesla")
        self.company_settings = company_config.get("settings", {})
        self.max_tier_depth = self.company_settings.get("max_tier_depth", 4)
        
        self.agents_config = load_yaml_config("config/agents.yaml")
        self.tasks_config = load_yaml_config("config/tasks.yaml")
        
        self._format_configs()
    
    def _format_configs(self):
        """Format agent and task configs with company name.
        
        Uses str.replace() instead of .format() to avoid conflicts with
        other curly-brace placeholders like {scenario_id} that are
        interpolated by CrewAI at runtime.
        """
        for agent_key, agent_config in self.agents_config.items():
            if isinstance(agent_config, dict):
                for key in ["role", "goal", "backstory"]:
                    if key in agent_config and isinstance(agent_config[key], str):
                        agent_config[key] = agent_config[key].replace(
                            "{company_name}", self.company_name
                        )
        
        for task_key, task_config in self.tasks_config.items():
            if isinstance(task_config, dict):
                for key in ["description", "expected_output"]:
                    if key in task_config and isinstance(task_config[key], str):
                        task_config[key] = task_config[key].replace(
                            "{company_name}", self.company_name
                        )

    @agent
    def disruption_monitoring_agent(self) -> Agent:
        from agents.disruption_monitoring_agent import DisruptionMonitoringAgent
        return DisruptionMonitoringAgent(
            company_name=self.company_name,
            config=self.agents_config["disruption_monitoring_agent"],
            tools=[DisruptionAnalysisTool()],
            verbose=True
        )

    @agent
    def kg_query_agent(self) -> Agent:
        from agents.kg_query_agent import KGQueryAgent
        from tools.kg_orchestration_tools import build_and_save_kg_tool
        return KGQueryAgent(
            config=self.agents_config["kg_query_agent"],
            tools=[wrap_tool(build_and_save_kg_tool)],
            verbose=True
        )

    @agent
    def product_search_agent(self) -> Agent:
        from agents.product_search_agent import ProductSearchAgent
        return ProductSearchAgent(
            config=self.agents_config["product_search_agent"],
            tools=[
                wrap_tool(openai_search_tool),
                wrap_tool(calculate_tier_struct),
                wrap_tool(resolve_entity_struct)
            ],
            verbose=True
        )
    
    @agent
    def risk_manager_agent(self) -> Agent:
        """Risk Manager orchestrates the deterministic risk calculation tool.
        
        The tool performs the heavy computation (same logic as ground truth
        generation). The agent's role is to invoke the tool with the correct
        scenario identifier; all data is loaded from disk via file-based
        data handoff.
        """
        from tools.tier1_comprehensive_risk_tool import tier1_comprehensive_risk_tool
        return Agent(
            config=self.agents_config["risk_manager_agent"],
            tools=[wrap_tool(tier1_comprehensive_risk_tool)],
            verbose=True
        )

    @agent
    def chief_supply_chain_agent(self) -> Agent:
        return ChiefSupplyChainAgent(
            config=self.agents_config["chief_supply_chain_agent"],
            tools=[],
            verbose=True
        )

    @agent
    def sourcing_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["sourcing_agent"],
            tools=[wrap_tool(openai_search_tool)],
            verbose=True
        )

    @task
    def scrape_news_task(self) -> Task:
        task_config = self.tasks_config["scrape_news_task"].copy()
        return Task(
            config=task_config,
            agent=self.disruption_monitoring_agent(),
        )

    @task
    def analyze_disruptions_task(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_disruptions_task"],
            agent=self.disruption_monitoring_agent()
        )

    @task
    def kg_query_task(self) -> Task:
        return Task(
            config=self.tasks_config["kg_query_task"],
            agent=self.kg_query_agent()
        )

    @task
    def product_search_task(self) -> Task:
        return Task(
            config=self.tasks_config["product_search_task"],
            agent=self.product_search_agent()
        )
    
    @task
    def risk_manager_task(self) -> Task:
        return Task(
            config=self.tasks_config["risk_manager_task"],
            agent=self.risk_manager_agent()
        )

    @task
    def decision_making_task(self) -> Task:
        return Task(
            config=self.tasks_config["decision_making_task"],
            agent=self.chief_supply_chain_agent()
        )

    @task
    def sourcing_task(self) -> Task:
        return Task(
            config=self.tasks_config["sourcing_task"],
            agent=self.sourcing_agent()
        )

    @crew
    def crew(self) -> Crew:
        """Build and return the crew.
        
        Set DISABLE_PRODUCT_AGENTS=1 to run in evaluation mode (core
        agents only: Disruption Monitor, KG Query, Risk Manager, CSCO).
        """
        import os
        disable_product_agents = os.environ.get("DISABLE_PRODUCT_AGENTS", "0") == "1"
        
        if disable_product_agents:
            agents_list = [
                self.disruption_monitoring_agent(),
                self.kg_query_agent(),
                self.risk_manager_agent(),
                self.chief_supply_chain_agent()
            ]
            tasks_list = [
                self.scrape_news_task(),
                self.analyze_disruptions_task(),
                self.kg_query_task(),
                self.risk_manager_task(),
                self.decision_making_task()
            ]
        else:
            agents_list = [
                self.disruption_monitoring_agent(),
                self.kg_query_agent(),
                self.product_search_agent(),
                self.risk_manager_agent(),
                self.chief_supply_chain_agent(),
                self.sourcing_agent()
            ]
            tasks_list = [
                self.scrape_news_task(),
                self.analyze_disruptions_task(),
                self.kg_query_task(),
                self.product_search_task(),
                self.risk_manager_task(),
                self.decision_making_task(),
                self.sourcing_task()
            ]
        
        return Crew(
            agents=agents_list,
            tasks=tasks_list,
            process=Process.sequential,
            verbose=True,
            full_output=True,
            planning=False
        )
