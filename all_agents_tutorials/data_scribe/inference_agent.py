import os
import logging
from dotenv import load_dotenv
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor, create_openai_functions_agent

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InferenceAgent:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.db = os.getenv("DATABASE")

        if not all([self.openai_api_key, self.db]):
            raise ValueError("Missing required environment variables: OPENAI_API_KEY, DATABASE")

        # Initialize DB
        self.dbEngine = SQLDatabase.from_uri(f"sqlite:///{self.db}")

        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0)

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

    def run_query(self, q: str) -> str:
        """Execute a raw SQL query"""
        try:
            return self.dbEngine.run(q)
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return f"Error executing query: {str(e)}"

    def create_chat_prompt(self) -> ChatPromptTemplate:
        """Create the chat prompt template"""
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a database inference expert for a SQLite database named {db_name}.
            Your job is to answer questions by querying the database and providing clear, accurate results.

            Rules:
            1. ONLY execute queries that retrieve data
            2. DO NOT provide analysis or recommendations
            3. Format responses as:
               Query Executed: [the SQL query used]
               Results: [the query results]
               Summary: [brief factual summary of the findings]
            4. Keep responses focused on the data only
            """
        )

        human_message = HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}")
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def query(self, text: str, **kwargs) -> str:
        """
        Process a query about the database.

        Args:
            text (str): The query text
            **kwargs: Additional arguments (like state)

        Returns:
            str: The query results and summary
        """
        try:
            logger.info(f"Processing inference query: {text}")

            result = self.agent_executor.invoke({
                "input": text,
                "db_name": self.db
            })

            return result["output"]

        except Exception as e:
            logger.error(f"Error in inference query: {str(e)}", exc_info=True)
            return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    agent = InferenceAgent()
    print("\nTesting direct query:")
    response = agent.query("How many albums are in the database?")
    print(response)