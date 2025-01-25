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
       # Refined prompt for better company info extraction
       prompt = f"Research the company or industry '{company_name_or_sector}' and provide a detailed summary. " \
                "Include its primary business model, key products or services, target markets, " \
                "strategic goals, and any relevant industry trends or news that are impacting the company. " \
                "Focus on providing specific and useful details."
       return self.agent.run(prompt)
   
   def generate_use_cases(self, company_summary):
       # Refined prompt to encourage diverse and creative AI/ML use cases
       prompt = f"Given the following company summary, brainstorm innovative AI/ML use cases for the company. " \
                "Think beyond standard applications; consider areas such as customer engagement, " \
                "data-driven innovation, AI-powered product offerings, supply chain, and R&D. " \
                "Provide detailed, actionable, and creative use cases based on the company’s context:\n\n{company_summary}\n\n" \
                "Ensure these use cases align with emerging trends in AI and ML."
       response = self.agent.run(prompt)
       use_cases = response.strip().splitlines()
       return list(set(use_cases))  # Remove duplicates from the generated use cases
   
   def find_relevant_resources(self, use_cases):
       resources = {}
       for use_case in use_cases:
           # Refined prompt for searching resources
           prompt = f"Find datasets, articles, tools, or libraries related to this AI/ML use case: \n\n{use_case}\n\n" \
                    "Search platforms like Kaggle, Hugging Face, GitHub, and Google Scholar. " \
                    "Provide links to the most useful resources, including open datasets, codebases, or research papers."
           response = self.agent.run(prompt)
           resource_links = response.strip().splitlines()
           resources[use_case] = resource_links
       return resources
   
   def propose_genai_solutions(self, use_cases, company_summary):
       # Refined prompt for GenAI solution proposal
       prompt = f"Based on the following company summary and AI/ML use cases, propose detailed GenAI solutions that the company could adopt. " \
                "Consider the company's strategic goals and current operations. " \
                "Provide examples of how generative AI could be used to drive growth, improve efficiency, " \
                "enhance customer experience, or optimize operations:\n\n" \
                f"**Company Summary:**\n{company_summary}\n\n" \
                f"**Use Cases:**\n{', '.join(use_cases)}\n\n" \
                "Propose innovative and unique solutions that can have a direct impact on the company."
       response = self.agent.run(prompt)
       genai_solutions = response.strip().splitlines()
       return list(set(genai_solutions))  # Remove duplicates from the proposed solutions


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

       # Step 1: Gather company information
       company_info = industry_research_agent.gather_information(company_name)
       st.subheader("Company Information")
       st.write(company_info)

       # Step 2: Generate AI/ML use cases
       generated_use_cases = industry_research_agent.generate_use_cases(company_info)
       st.subheader("Generated AI/ML Use Cases")
       st.markdown("Here are some potential AI/ML use cases for the company:")
       for case in generated_use_cases:
           st.markdown(f"- {case}")

       # Step 3: Collect relevant resources
       resources = industry_research_agent.find_relevant_resources(generated_use_cases)
       st.subheader("Collected Resources")
       st.markdown("Here are some relevant resources for the use cases:")
       for use_case, links in resources.items():
           st.markdown(f"**{use_case}:**")
           for link in links:
               st.markdown(f"- {link}")

       # Step 4: Propose GenAI solutions
       genai_solutions = industry_research_agent.propose_genai_solutions(generated_use_cases, company_info)
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
