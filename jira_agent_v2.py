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
    model="gemini-2.5-pro-exp-03-25",
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

# Define tools using @tool decorator
@tool
def create_issue(input_str: str) -> str:
    """Create a new Jira issue.
    Input should be in the format: project_key, summary, description
    separated by | characters."""
    parts = input_str.split('|')
    if len(parts) < 2:
        return "Error: Input should contain at least project_key and summary separated by |"
    
    project_key = parts[0].strip()
    summary = parts[1].strip()
    description = parts[2].strip() if len(parts) > 2 else ""
    
    try:
        # Use the issue_create method directly
        result = jira.issue_create(project_key, summary, description)
        return f"Successfully created issue: {result}"
    except Exception as e:
        return f"Error creating issue: {str(e)}"

@tool
def search_jira(input_str: str) -> str:
    """Search for Jira issues.
    Input should be a JQL query or simple search term."""
    try:
        # Use the search method directly
        result = jira.search(input_str)
        return f"Search results: {result}"
    except Exception as e:
        return f"Error searching issues: {str(e)}"

@tool
def get_project_info(project_key: str) -> str:
    """Get information about a specific Jira project."""
    try:
        # Use the project method directly
        result = jira.project(project_key)
        return f"Project information: {result}"
    except Exception as e:
        return f"Error retrieving project information: {str(e)}"

# Define the tools list
tools = [create_issue, search_jira, get_project_info]

# Create the ReAct agent
jira_agent = create_react_agent(model, tools)

def interact_with_jira(query: str):
    """Process a natural language query about Jira and execute the appropriate action."""
    response = jira_agent.invoke({"messages": [("human", query)]})
    return response['messages'][-1].content

# Helper function to create a Jira issue with error handling
def create_jira_issue(summary, description=None, project=PROJECT_KEY):
    """Helper function to create a Jira issue with proper error handling"""
    try:
        prompt = f"Create a new Jira issue in project {project} with summary '{summary}'"
        if description:
            prompt += f" and description '{description}'"
        
        return interact_with_jira(prompt)
    except Exception as e:
        print(f"Error creating Jira issue: {str(e)}")
        return f"Failed to create issue: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Example 1: Create a new issue
    result = create_jira_issue("Make more fried rice", "Remember to make more fried rice.")
    print(result)
    
    # Example 2: Direct agent interaction
    query = f"Show me all issues in project {PROJECT_KEY}"
    result = interact_with_jira(query)
    print(result)