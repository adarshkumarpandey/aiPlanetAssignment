from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import csv
from googleapiclient.discovery import build


# Load environment variables from .env file
load_dotenv()


# Access environment variables
api_key = os.getenv('openAI_api_key')
google_api_key = os.getenv('GOOGLE_API_KEY')
google_cse_id = os.getenv('GOOGLE_CSE_ID')


# Implement the google_search function using Google Custom Search API
def google_search(query):
    # Build the service object to interact with Google's API
    service = build("customsearch", "v1", developerKey=google_api_key)
    
    # Perform the search query
    res = service.cse().list(q=query, cx=google_cse_id).execute()
    
    # Extract the search results from the response
    search_results = []
    if 'items' in res:
        for item in res['items']:
            search_results.append(f"{item['title']}: {item['link']}")
    else:
        search_results.append("No results found.")
    
    return "\n".join(search_results)


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
company_name = "Waitrose"
company_info = industry_research_agent.gather_information(company_name)


# Usecase Generation using company information
class UseCaseGenerationAgent:
    def __init__(self, llm, tools):
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent_type="zero-shot-react-description",  # Common agent type for this use case
            verbose=True  # For debugging purposes
        )

    def generate_use_cases(self, company_summary):
        prompt = f"Based on the following company summary, brainstorm potential AI/ML use cases: \n\n{company_summary}\n\n" \
                 "Consider use cases related to operations, customer experience, product development, and other relevant areas. " \
                 "Be creative and explore innovative applications."
        
        response = self.agent.run(prompt)

        use_cases = response.strip().splitlines()
        return use_cases if use_cases != [''] else ["NA"]


# Initialize the agent with the LLM and tools
use_case_agent = UseCaseGenerationAgent(llm, tools)

# Generate AI/ML use cases for the given company summary
generated_use_cases = use_case_agent.generate_use_cases(company_info)


# Resource Collection Agent
class ResourceCollectionAgent:
    def __init__(self, llm, tools):
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent_type="zero-shot-react-description",  # Agent type for zero-shot reasoning
            verbose=True  # Set to True for debugging
        )

    def find_relevant_resources(self, use_cases):
        resources = {}
        for use_case in use_cases:
            prompt = f"Find relevant datasets and resources (e.g., libraries, tools, articles) for the following AI/ML use case: \n\n{use_case}\n\n" \
                     "Search on platforms like Kaggle, Hugging Face, GitHub, and Google Scholar. " \
                     "Provide links to the most relevant resources."
            response = self.agent.run(prompt)
            resource_links = response.strip().splitlines()
            resources[use_case] = resource_links if resource_links != [''] else ["NA"]
        return resources


# Solution Proposal Agent
class SolutionProposalAgent:
    def __init__(self, llm, tools):
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent_type="zero-shot-react-description",  # Agent type for zero-shot reasoning
            verbose=True,  # Set to True for debugging
            handle_parsing_errors=True  # Allow handling parsing errors automatically
        )

    def propose_genai_solutions(self, use_cases, company_summary):
        # Adjusting the prompt to make the output cleaner
        prompt = f"Based on the following company summary and AI/ML use cases, propose specific GenAI solutions:\n" \
                 f"Company Summary:\n{company_summary}\n" \
                 f"Use Cases:\n{', '.join(use_cases)}\n" \
                 "Please list the proposed GenAI solutions in bullet points without any extra sentences or context. Only the solutions, each on a new line."

        # Run the agent to generate the response
        response = self.agent.run(prompt)
        
        # Clean up the response if needed, remove any unwanted starting or ending phrases
        if response:
            clean_response = response.strip()
            # Remove any prefix like "I now know the final answer:"
            clean_response = clean_response.replace("I now know the final answer:", "").strip()
            
            # If the response still contains data, split it into lines for better readability
            genai_solutions = clean_response.splitlines() if clean_response else ["NA"]
        else:
            genai_solutions = ["NA"]

        return genai_solutions


# Initialize agents
resource_collection_agent = ResourceCollectionAgent(llm, tools)
solution_proposal_agent = SolutionProposalAgent(llm, tools)

# Collect relevant resources for the use cases
resources = resource_collection_agent.find_relevant_resources(generated_use_cases)

# Propose GenAI solutions based on the use cases and company summary
genai_solutions = solution_proposal_agent.propose_genai_solutions(generated_use_cases, company_info)


# Prepare the CSV data
csv_data = []
company_name = "Waitrose"  # Example, could be dynamic

# Usecases: Joining each item with '\n' for better readability
use_cases_str = "\n".join(generated_use_cases)

# Resource Collections: Formatted with each use case and its associated links
resources_str = "\n".join([f"{use_case}:\n- " + "\n- ".join(links) for use_case, links in resources.items()])

# GenAI Solutions: List of solutions, each on a new line
solutions_str = "\n".join(genai_solutions)

# Append the formatted data
csv_data.append([company_name, use_cases_str, resources_str, solutions_str])

# Write to CSV with proper formatting
csv_file = "company_research_output.csv"
csv_header = ["Company_name", "Usecases", "Resource_Collections", "Solution_Proposed"]

# Write header and data to the CSV file
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)  # Write the header
    writer.writerows(csv_data)  # Write the collected data

print(f"Data has been written to {csv_file}")
