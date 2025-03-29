import os
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities.jira import JiraAPIWrapper

# Environment variables
GEM_API_KEY = os.environ.get("GEM_API_KEY")
JIRA_USERNAME = os.environ.get("JIRA_USERNAME")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN")
JIRA_INSTANCE_URL = os.environ.get("JIRA_INSTANCE_URL")
JIRA_CLOUD = os.environ.get("JIRA_CLOUD", "False")
PROJECT_KEY = os.environ.get("PROJECT_KEY")

# Validate environment variables
required_vars = ["GEM_API_KEY", "JIRA_USERNAME", "JIRA_API_TOKEN", "JIRA_INSTANCE_URL", "PROJECT_KEY"]
missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize LLM
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro-exp-03-25",  # Using Gemini 2.5 experimental
    google_api_key=GEM_API_KEY,
    temperature=0.1
)

# Initialize Jira API wrapper
jira = JiraAPIWrapper(
    jira_username=JIRA_USERNAME,
    jira_api_token=JIRA_API_TOKEN,
    jira_instance_url=JIRA_INSTANCE_URL,
    jira_cloud=(JIRA_CLOUD.lower() == "true")
)

# Define custom tools using @tool decorator
@tool
def create_jira_ticket(input_str: str) -> str:
    """Create a new Jira ticket.
    Input should be in the format: project_key, summary, description
    separated by | characters."""
    parts = input_str.split('|')
    if len(parts) < 2:
        return "Error: Input should contain at least project_key and summary separated by |"
    
    project_key = parts[0].strip()
    summary = parts[1].strip()
    description = parts[2].strip() if len(parts) > 2 else ""
    
    try:
        # Using the correct method from JiraAPIWrapper
        result = jira.create_ticket(project_key, summary, description)
        return f"Successfully created ticket: {result}"
    except Exception as e:
        return f"Error creating ticket: {str(e)}"

@tool
def get_all_tickets_for_project(project_key: str) -> str:
    """Get all tickets for a specific project."""
    try:
        # Using the correct method from JiraAPIWrapper
        return jira.get_tickets_by_project(project_key)
    except Exception as e:
        return f"Error retrieving tickets: {str(e)}"

@tool
def jql_search(query: str) -> str:
    """Search for tickets using JQL (Jira Query Language)."""
    try:
        # Using the correct method from JiraAPIWrapper
        return jira.run_jql_query(query)
    except Exception as e:
        return f"Error running JQL query: {str(e)}"

@tool
def get_ticket_by_id(ticket_id: str) -> str:
    """Get details of a specific ticket by its ID."""
    try:
        # Using the correct method from JiraAPIWrapper
        return jira.get_ticket(ticket_id)
    except Exception as e:
        return f"Error retrieving ticket: {str(e)}"

# Define the tools list
tools = [create_jira_ticket, get_all_tickets_for_project, jql_search, get_ticket_by_id]

# Create the ReAct agent
jira_agent = create_react_agent(model, tools)

def interact_with_jira(query: str):
    """Process a natural language query about Jira and execute the appropriate action."""
    response = jira_agent.invoke({"messages": [("human", query)]})
    return response['messages'][-1].content

# Helper function to create a Jira ticket with error handling
def create_new_jira_ticket(summary, description=None, project=PROJECT_KEY):
    """Helper function to create a Jira ticket with proper error handling"""
    try:
        prompt = f"Create a new Jira ticket in project {project} with summary '{summary}'"
        if description:
            prompt += f" and description '{description}'"
        
        return interact_with_jira(prompt)
    except Exception as e:
        print(f"Error creating Jira ticket: {str(e)}")
        return f"Failed to create ticket: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Example 1: Create a new ticket
    result = create_new_jira_ticket("Make more fried rice", "Remember to make more fried rice.")
    print(result)
    
    # Example 2: Direct agent interaction
    query = f"Show me all issues in project {PROJECT_KEY}"
    result = interact_with_jira(query)
    print(result)