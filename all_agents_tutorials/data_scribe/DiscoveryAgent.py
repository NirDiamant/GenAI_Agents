import os
import logging
from dotenv import load_dotenv
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Any, Dict, Optional, List
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool

import json

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiscoveryAgent:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.db = os.getenv("DATABASE")

        if not all([self.openai_api_key, self.db]):
            raise ValueError("Missing required environment variables: OPENAI_API_KEY, DATABASE")

        # Initialize DB
        self.dbEngine = SQLDatabase.from_uri(f"sqlite:///{self.db}")

        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4")

        # Create toolkit and tools
        self.toolkit = SQLDatabaseToolkit(db=self.dbEngine, llm=self.llm)
        self.tools = self.toolkit.get_tools()

        # Add results formatting tool
        self.tools.append(
            Tool(
                name="RESULTS",
                func=self.format_results_for_graph,
                description="Use this function to format your final results for graphing. Pass your data as a string."
            )
        )

        # Create prompt
        self.chat_prompt = self.create_chat_prompt()

        # Create agent and executor
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            prompt=self.chat_prompt,
            tools=self.tools
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15
        )

        self.test_connection()

    def test_connection(self):
        self.show_tables()

    def run_query(self, q):
        return self.dbEngine.run(q)

    def show_tables(self):
        q = '''
            SELECT
                name,
                type
            FROM sqlite_master
            WHERE type IN ("table","view");
            '''
        return self.run_query(q)

    def create_chat_prompt(self):
        system_message = SystemMessagePromptTemplate.from_template(
            """
            You are an AI assistant for querying a SQLLite database named {db_name}.
            Your responses should be formatted for readability, using line breaks and bullet points where appropriate.
            When listing items, use a numbered or bulleted list. Always strive for clarity and conciseness in your responses.
            When querying for table names, use the SHOW TABLES command. To get information about a table's structure, use the
            DESCRIBE command followed by the table name. When providing SQL queries, do not wrap them in code blocks or backticks; instead, provide the raw SQL query directly.
            """
        )

        human_message = HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}")
        return ChatPromptTemplate.from_messages([system_message, human_message])

    @staticmethod
    def format_results_for_graph(data):
        try:
            parsed_data = json.loads(data)
            return json.dumps({"graph_data": parsed_data})
        except json.JSONDecodeError:
            return json.dumps({"graph_data": []})

    def process_input(self, user_input: str) -> dict:
        """Process a single user input and return the response"""
        try:
            logger.info(f"Processing user input: {user_input}")
            response = self.agent_executor.invoke({"input": user_input, "db_name": self.db})
            logger.info("Agent response provided successfully")
            return response
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def handle_response(self, agent: Any, response: Dict[str, Any], graph_requested: bool) -> None:
        """Handle the agent's response and any special formatting"""
        if 'error' in response:
            print(f"An error occurred: {response['error']}")
            print("Please check the logs for more detailed information.")
            return

        print("\nAgent:")
        print(response['output'])

agent = DiscoveryAgent()

print (agent.show_tables())

agent.process_input("Is there a table called albums?")
agent.process_input("Is there a table called peoples?")
agent.process_input("How many tables are there in the database?")