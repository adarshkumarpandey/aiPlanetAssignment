import streamlit as st
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import csv
import pandas as pd
from googleapiclient.discovery import build

# Load environment variables from .env file
load_dotenv()

# Access environment variables
api_key = os.getenv('openAI_api_key')
google_api_key = os.getenv('GOOGLE_API_KEY')
google_cse_id = os.getenv('GOOGLE_CSE_ID')

# Example search function (replace with actual Google search or another tool)
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
        return use_cases

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
            resources[use_case] = resource_links
        return resources

class SolutionProposalAgent:
    def __init__(self, llm, tools):
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent_type="zero-shot-react-description",  # Agent type for zero-shot reasoning
            verbose=True
        )

    def propose_genai_solutions(self, use_cases, company_summary):
        prompt = f"Based on the following company summary and AI/ML use cases, propose specific GenAI solutions: \n\n" \
                 f"**Company Summary:**\n{company_summary}\n\n" \
                 f"**Use Cases:**\n{'\n'.join(use_cases)}\n\n" \
                 "Consider solutions like document search, automated report generation, AI-powered chat systems, and other relevant applications. " \
                 "Explain how each solution can benefit the company."

        response = self.agent.run(prompt)
        genai_solutions = response.strip().splitlines()
        return genai_solutions

# Streamlit UI
st.title("AI/ML Use Case Generation and Solution Proposal")

company_name = st.text_input("Enter Company Name", "")

if st.button("Generate Results"):
    if company_name:
        llm = OpenAI(openai_api_key=api_key)

        tools = [
            Tool(
                name="Search",
                func=google_search,
                description="For when you need to search for something."
            )
        ]

        # Initialize agents
        industry_research_agent = IndustryResearchAgent(llm, tools)
        use_case_agent = UseCaseGenerationAgent(llm, tools)
        resource_collection_agent = ResourceCollectionAgent(llm, tools)
        solution_proposal_agent = SolutionProposalAgent(llm, tools)

        # Step 1: Gather company information
        company_info = industry_research_agent.gather_information(company_name)
        st.subheader("Company Information")
        st.write(company_info)

        # Step 2: Generate AI/ML use cases
        generated_use_cases = use_case_agent.generate_use_cases(company_info)
        st.subheader("Generated AI/ML Use Cases")
        st.markdown("Here are some potential AI/ML use cases for the company:")
        for case in generated_use_cases:
            st.markdown(f"- {case}")

        # Step 3: Collect relevant resources
        resources = resource_collection_agent.find_relevant_resources(generated_use_cases)
        st.subheader("Collected Resources")
        st.markdown("Here are some relevant resources for the use cases:")
        for use_case, links in resources.items():
            st.markdown(f"**{use_case}:**")
            for link in links:
                st.markdown(f"- {link}")

        # Step 4: Propose GenAI solutions
        genai_solutions = solution_proposal_agent.propose_genai_solutions(generated_use_cases, company_info)
        st.subheader("Proposed GenAI Solutions")
        st.markdown("Here are some potential GenAI solutions:")
        for solution in genai_solutions:
            st.markdown(f"- {solution}")

        # Export results to CSV
        output_data = {
            "Company_name": company_name,
            "Usecases": "\n".join(generated_use_cases),
            "Resource_Collections": "\n".join([f"{use_case}: {', '.join(resources[use_case])}" for use_case in resources]),
            "Solution_Proposed": "\n".join(genai_solutions)
        }

        # Save data to CSV and allow download
        df = pd.DataFrame([output_data])
        csv = df.to_csv(index=False)
        st.download_button("Download Results as CSV", csv, file_name=f"{company_name}_AI_Solutions.csv")
    
    else:
        st.error("Please enter a company name.")
