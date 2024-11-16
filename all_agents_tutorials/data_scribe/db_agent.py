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

class DBAgent:
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
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=15
        )

        self.test_connection()

    def test_connection(self):
        """Test database connection on initialization"""
        try:
            self.show_tables()
            logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise

    def run_query(self, q: str) -> str:
        """Execute a raw SQL query"""
        try:
            return self.dbEngine.run(q)
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return f"Error executing query: {str(e)}"

    def show_tables(self) -> str:
        """Show all tables in the database"""
        q = '''
            SELECT
                name,
                type
            FROM sqlite_master
            WHERE type IN ("table","view");
            '''
        return self.run_query(q)

    def create_chat_prompt(self) -> ChatPromptTemplate:
        """Create the chat prompt template"""
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a database query expert for a SQLite database named {db_name}.
            Analyze the user's request and provide:
            1. The appropriate SQL query
            2. The query results
            3. A brief explanation

            Format your response exactly as:
            Query Executed: [the SQL query used]
            Results: [the query results]
            Summary: [brief explanation of the findings]"""
        )

        human_message = HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}")
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def process_input(self, user_input: str, **kwargs) -> dict:
        """Internal method to process user input through the agent"""
        try:
            if 'state' in kwargs:
                logger.info("Current State in DBAgent:")
                logger.info(json.dumps(kwargs['state'], indent=2))

            logger.info(f"Processing user input: {user_input}")
            response = self.agent_executor.invoke({
                "input": user_input,
                "db_name": self.db
            })
            logger.info("Agent response provided successfully")
            return response
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
            return {
                "output": f"""
                Query Executed: ERROR
                Results: Query processing failed
                Summary: An error occurred: {str(e)}
                """
            }

    def query(self, text: str, **kwargs) -> str:
        """
        Process a query about the database.

        Args:
            text (str): The query text
            **kwargs: Additional arguments (like state)

        Returns:
            str: The response from the agent
        """
        state = kwargs.get('state', {})

        # Execute the agent
        result = self.agent_executor.invoke(
            {
                "input": text,
                "state": state,
                "db_name": self.db
            }
        )

        return result["output"]

# For testing purposes
if __name__ == "__main__":
    agent = DBAgent()
    print("\nTesting direct query:")
    response = agent.query("How many albums are in the database?")
    print(response)