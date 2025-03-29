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

# Create a custom wrapper around the JiraAPIWrapper that provides the methods we need
class CustomJiraWrapper:
    def __init__(self, jira_wrapper):
        self.jira = jira_wrapper
    
    def create_new_issue(self, project_key, summary, description=""):
        """Create a new Jira issue using the run method."""
        try:
            # Directly construct the input string expected by the run method
            input_str = f"Make a new issue in project {project_key} with summary '{summary}'"
            if description:
                input_str += f" and description '{description}'"
            
            result = self.jira.run(input_str)
            return f"Created issue: {result}"
        except Exception as e:
            return f"Error creating issue: {str(e)}"
    
    def get_all_project_issues(self, project_key):
        """Get all issues for a project using the run method."""
        try:
            input_str = f"Find all issues in project {project_key}"
            return self.jira.run(input_str)
        except Exception as e:
            return f"Error getting issues: {str(e)}"
    
    def search_jql(self, jql_query):
        """Search issues using JQL via the run method."""
        try:
            input_str = f"Search for issues with JQL: {jql_query}"
            return self.jira.run(input_str)
        except Exception as e:
            return f"Error searching issues: {str(e)}"
    
    def get_issue_by_key(self, issue_key):
        """Get a specific issue by key using the run method."""
        try:
            input_str = f"Get issue {issue_key}"
            return self.jira.run(input_str)
        except Exception as e:
            return f"Error getting issue: {str(e)}"

# Create the custom wrapper
custom_jira = CustomJiraWrapper(jira)

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
    
    return custom_jira.create_new_issue(project_key, summary, description)

@tool
def list_project_issues(project_key: str) -> str:
    """List all issues for a specific project."""
    return custom_jira.get_all_project_issues(project_key)

@tool
def run_jql_query(query: str) -> str:
    """Run a JQL (Jira Query Language) query to search for issues."""
    return custom_jira.search_jql(query)

@tool
def get_issue(issue_key: str) -> str:
    """Get details of a specific issue by its key."""
    return custom_jira.get_issue_by_key(issue_key)

# Define the tools list
tools = [create_issue, list_project_issues, run_jql_query, get_issue]

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