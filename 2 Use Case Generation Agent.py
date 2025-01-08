from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI  # Replace with your preferred LLM
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from environment variables
api_key = os.getenv('openAI_api_key')

# Example external tool (you can replace this with actual functionality, e.g., web search)
def example_tool(query):
    # This is just a mock tool for demonstration. Replace it with your actual tool logic.
    return f"Mock tool result for query: {query}"

class UseCaseGenerationAgent:
    def __init__(self, llm, tools):
        """
        Initializes the UseCaseGenerationAgent with an LLM and a set of tools.

        Args:
            llm: The language model (e.g., OpenAI LLM) for generating responses.
            tools: A list of tools that can be used during agent execution.
        """
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent_type="zero-shot-react-description",  # Common agent type for this use case
            verbose=True  # For debugging purposes
        )

    def generate_use_cases(self, company_summary):
        """
        Generates potential AI/ML use cases based on the company summary.

        Args:
            company_summary: A summary of the company, its industry, and key information.

        Returns:
            A list of generated AI/ML use cases.
        """
        prompt = f"Based on the following company summary, brainstorm potential AI/ML use cases: \n\n{company_summary}\n\n" \
                 "Consider use cases related to operations, customer experience, product development, and other relevant areas. " \
                 "Be creative and explore innovative applications."
        
        # Use the agent to generate a response
        response = self.agent.run(prompt)

        # Process the response (strip extra whitespace and split by lines for individual use cases)
        use_cases = response.strip().splitlines()
        return use_cases

# Example tool list (you can define more complex tools here)
tools = [
    Tool(
        name="ExampleTool", 
        func=example_tool,
        description="A mock tool to demonstrate agent integration"
    ),
]

# Initialize the OpenAI LLM with the API key
llm = OpenAI(openai_api_key=api_key)  # Ensure your OpenAI API key is in .env

# Initialize the agent with the LLM and tools
use_case_agent = UseCaseGenerationAgent(llm, tools)

# Example company summary
company_summary = """
Tesla is an American electric vehicle and clean energy company. 
Founded in 2003, Tesla designs and manufactures electric vehicles, battery energy storage from home to grid-scale, 
solar panels and solar roof tiles, and related products and services. 
Tesla's mission is to accelerate the world's transition to sustainable energy. 
"""

# Generate AI/ML use cases for the given company summary
generated_use_cases = use_case_agent.generate_use_cases(company_summary)

# Output the generated use cases
print("Generated Use Cases:")
for use_case in generated_use_cases:
    print(f"- {use_case}")

#Outout
"""
 Finished chain.
Generated Use Cases:
- Some potential AI/ML use cases for Tesla could include using machine learning algorithms to 
analyze data for product development and using chatbots for personalized customer recommendations. 
Other potential use cases could include using NLP to analyze customer feedback and sentiment, using 
image recognition for quality control in manufacturing, and using predictive analytics for optimizing
 operations and energy production.

"""
