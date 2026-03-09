# main.py

import io
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from crew import SupplyChainCrew
from tools.neo4j_setup import shutdown


def save_agent_output(agent_name: str, agent_output: str, directory: str = "output"):
    """Save the given agent's output to a separate text file."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{agent_name}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(agent_output)


def capture_stdout(func, *args, **kwargs):
    """Capture all printed output during a function call."""
    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer
    try:
        result = func(*args, **kwargs)
        printed_output = buffer.getvalue()
    finally:
        sys.stdout = sys_stdout
    return result, printed_output


def run():
    from crew import load_company_config
    company_config = load_company_config()
    company_name = company_config.get("name", "Tesla")
    
    website_url = "https://www.britannica.com/event/2022-Russian-invasion-of-Ukraine"
    print(f"Running disruption analysis for {company_name} with URL: {website_url}")

    try:
        crew_instance = SupplyChainCrew(company_name=company_name).crew()

        final_output, full_thinking_log = capture_stdout(
            crew_instance.kickoff,
            inputs={"website_url": website_url, "company_name": company_name}
        )

        save_agent_output("full_thinking_log", full_thinking_log, directory="output")

        agent_outputs = {}
        for task in crew_instance.tasks:
            agent = task.agent
            if agent:
                agent_name = agent.role
                task_output = str(task.output) if task.output else "No output generated."
                if agent_name not in agent_outputs:
                    agent_outputs[agent_name] = []
                agent_outputs[agent_name].append(task_output)

        for agent_name, outputs in agent_outputs.items():
            combined_output = "\n\n".join(outputs)
            save_agent_output(agent_name, combined_output, directory="output")

        save_agent_output("final_report", str(final_output), directory="output")

        print("\nFinal Supply Chain Risk Report:\n")
        print(final_output)

    except Exception as e:
        print(f"Error during execution: {e}")


if __name__ == "__main__":
    run()
