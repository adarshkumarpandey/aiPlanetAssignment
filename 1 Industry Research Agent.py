from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
api_key = os.getenv('openAI_api_key')

# Example search function (replace with actual Google search or another tool)
def google_search(query):
    # Implement the search logic (e.g., using an API or a scraping tool)
    return "Sample search result for: " + query

class IndustryResearchAgent:
    def __init__(self, llm, tools):
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent_type="zero-shot-react-description",  # Common agent type
            verbose=True  # For debugging purposes
        )

    def gather_information(self, company_name_or_sector):
        prompt = f"Research the company or industry '{company_name_or_sector}'. " \
                 "Find information like its industry, key offerings, strategic focus areas, " \
                 "and any relevant news or reports. Summarize the findings concisely."

        return self.agent.run(prompt)

# Define the LLM and tools
tools = [
    Tool(
        name="Search",
        func=google_search,
        description="For when you need to search for something.",
    ),
]

# Initialize the LLM
llm = OpenAI(openai_api_key=api_key)

# Initialize the agent
industry_research_agent = IndustryResearchAgent(llm, tools)

# Example usage
company_name = "Tesla"
company_info = industry_research_agent.gather_information(company_name)




#Output
"""
Finished chain.
Tesla is a company in the automotive and energy industry, with key offerings in electric vehicles,
 solar panels, and energy storage solutions. Their strategic focus areas include sustainable energy and 
 transportation, innovation, and expanding their market reach. Recent news includes record-breaking vehicle 
 deliveries and a potential partnership with a Chinese battery manufacturer.
"""