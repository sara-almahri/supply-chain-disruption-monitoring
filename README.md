# Supply Chain Disruption Monitoring Framework

An agentic AI framework for monitoring supply chain disruptions using large language models, knowledge graphs, and multi-agent orchestration.

## Overview

The framework coordinates four core agents in a sequential pipeline to process disruption signals from unstructured news articles and produce structured mitigation recommendations:

1. **Disruption Monitor** — extracts disruption type, affected countries, industries, and companies from news text using LLM-based entity extraction.
2. **Knowledge Graph Query Agent** — traces disrupted supply chain paths through a Neo4j knowledge graph up to Tier-4 depth.
3. **Risk Manager** — computes quantitative risk scores for affected Tier-1 suppliers using a deterministic, tool-grounded calculation.
4. **Chief Supply Chain Officer (CSCO)** — generates action recommendations (e.g., maintain standard, increase monitoring, activate contingency) based on risk scores.

A file-based data handoff mechanism passes only scenario identifiers between agents, with tools loading full data from disk. This reduces token usage and eliminates serialisation errors.

The multi-agent orchestration is built on [CrewAI](https://www.crewai.com/) ([GitHub](https://github.com/crewAIInc/crewAI)), a framework for orchestrating autonomous AI agents with defined roles, goals, and tools.

## Repository Structure

```
.
├── main.py                     # Entry point
├── crew.py                     # Multi-agent crew orchestration (CrewAI)
├── config/
│   ├── agents.yaml             # Agent role, goal, and backstory definitions
│   ├── tasks.yaml              # Task descriptions and expected outputs
│   └── company_config.yaml     # Target company configuration
├── agents/                     # Agent implementations
│   ├── disruption_monitoring_agent.py
│   ├── kg_query_agent.py
│   ├── chief_supply_chain_agent.py
│   └── ...
├── tools/                      # Tool implementations
│   ├── disruption_analysis_tool.py
│   ├── kg_orchestration_tools.py
│   ├── tier1_comprehensive_risk_tool.py
│   ├── neo4j_setup.py
│   └── ...
├── dataset/
│   └── supplychainKG.csv       # Supply chain knowledge graph data
├── scripts/
│   └── kg_ingestion.py         # Neo4j knowledge graph ingestion
├── evaluation/
│   ├── scenarios/              # 30 synthesised disruption scenarios
│   ├── ground_truth/           # Ground truth for all 30 scenarios
│   ├── evaluation_harness.py   # Multi-level evaluation framework
│   ├── generate_final_analysis.py  # Publication figure generation
│   ├── ws5_final_analysis.py   # Multi-model multi-run analysis
│   └── results/                # Experimental results
│       ├── baseline/           # GPT-4o: 30 scenarios × 5 runs
│       ├── gpt-4.1/            # gpt-4.1: 30 scenarios × 5 runs
│       ├── gpt-5-mini/         # gpt-5-mini: 28 scenarios × 5 runs
│       ├── prompt_sensitivity/  # Prompt paraphrase robustness
│       ├── noise_sensitivity/   # Input noise robustness
│       ├── fault_injection/     # Fault injection experiments
│       └── analysis/           # Summary reports and figures
├── requirements.txt
├── LICENSE
└── .env.example
```

## Evaluation Dataset

The evaluation benchmark consists of 30 synthesised disruption scenarios across three automotive manufacturers (Tesla, BMW, Mercedes-Benz), each with deterministic ground truth for:

- Disruption type and affected entities
- Disrupted supply chain paths
- Supplier risk scores
- Expected mitigation actions

All scenarios, ground truth, and experimental results (450 runs across three LLM models) are included in `evaluation/`.

## Key Results

Evaluated on 30 scenarios with 5 runs each (150 total per model):

| Metric | GPT-4o (baseline) | gpt-4.1 | gpt-5-mini |
|---|---|---|---|
| DM composite F1 | 0.913 ± 0.151 | 0.913 ± 0.115 | 0.926 ± 0.113 |
| KG strict success | 0.93 (28/30) | 0.93 (28/30) | 0.83 (26/28) |
| Risk strict success | 1.00 (30/30) | 1.00 (30/30) | 1.00 (28/28) |
| CSCO action accuracy | 0.918 ± 0.068 | 0.863 ± 0.185 | 0.878 ± 0.085 |
| Constrained success | 0.69 (22/30) | 0.65 (21/30) | 0.45 (14/28) |
| Mean latency | 102.5s | 137.2s | 1091.3s |

## Setup

### Prerequisites

- Python 3.9+
- [CrewAI](https://www.crewai.com/) — install via `pip install crewai` ([documentation](https://docs.crewai.com/))
- Neo4j database ([AuraDB](https://neo4j.com/cloud/aura/) or self-hosted)
- OpenAI API key

### Installation

```bash
git clone https://github.com/sara-almahri/supply-chain-disruption-monitoring.git
cd supply-chain-disruption-monitoring
pip install -r requirements.txt
```

> **Note:** `crewai` is included in `requirements.txt`. For the latest installation instructions, see the [CrewAI documentation](https://docs.crewai.com/).

### Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:
- `NEO4J_URI` — Neo4j connection URI
- `NEO4J_USERNAME` — Neo4j username
- `NEO4J_PASSWORD` — Neo4j password
- `OPENAI_API_KEY` — OpenAI API key

### Knowledge Graph Ingestion

Load the supply chain data into Neo4j:

```bash
python scripts/kg_ingestion.py
```

### Running the Framework

```bash
python main.py
```

To run in evaluation mode (core agents only, without product search and sourcing):

```bash
DISABLE_PRODUCT_AGENTS=1 python main.py
```

## Reproducing Evaluation Results

The evaluation harness can recompute all metrics from the included run data:

```bash
# Run the multi-level evaluation harness
python evaluation/evaluation_harness.py

# Regenerate multi-model analysis and figures
python evaluation/ws5_final_analysis.py

# Regenerate all publication figures
python evaluation/generate_final_analysis.py
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/). See [LICENSE](LICENSE) for details.
