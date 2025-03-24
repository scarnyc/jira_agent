import os
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

GEM_API_KEY = os.environ["GEM_API_KEY"]
JIRA_USERNAME = os.environ["JIRA_USERNAME"]
JIRA_INSTANCE_URL = os.environ["JIRA_INSTANCE_URL"]
JIRA_CLOUD = os.environ["JIRA_CLOUD"]
PROJECT_KEY = os.environ["PROJECT_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=GEM_API_KEY,
    temperature=0.1
)

jira = JiraAPIWrapper(
    jira_username=JIRA_USERNAME,
    jira_api_token=os.environ.get("JIRA_API_TOKEN"),
    jira_instance_url=JIRA_INSTANCE_URL,
    project_key=PROJECT_KEY
)

toolkit = JiraToolkit.from_jira_api_wrapper(jira)

agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run("make a new issue in project PW to remind me to make more fried rice")