import os
import logging
from dotenv import load_dotenv
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from typing import Any, Dict, Optional, List
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from pathlib import Path
import networkx as nx
from langchain_community.graphs.index_creator import GraphIndexCreator
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

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

        # Add results formatting tool and question answering tool
        self.tools.extend([
            Tool(
                name="RESULTS",
                func=self.format_results_for_graph,
                description="Use this function to format your final results for graphing. Pass your data as a string."
            ),
            Tool(
                name="ANSWER_QUESTION",
                func=self.answer_question,
                description="Use this function to answer general questions about the database content and structure."
            ),
            Tool(
                name="VISUALISE_SCHEMA",
                func=self.discover,
                description="Creates a visual graph representation of the database schema showing tables, columns, and their relationships."
            )
        ])

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
            Your responses should be formatted as json only.
            Always strive for clarity, terseness and conciseness in your responses.
            Return a json array with all the tables, using the example below:
            
            Example output:
            ```json
            [
                {{
                    tableName: [NAME OF TABLE RETURNED],
                    columns: [
                        {{
                            columnName: [COLUMN 1 NAME],
                            columnType: [COLUMN 1 TYPE]
                        }},
                        {{
                            columnName: [COLUMN 2 NAME],
                            columnType: [COLUMN 2 TYPE]
                        }}
                    ]
                }}
            ]
            ```
            
            ## mandatory
            only output json
            do not put any extra commentary 
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

    def discover(self, *args):
        """
        Creates a visual representation of the database schema.
        Ignores any additional arguments passed by the tool system.
        """
        prompt = "For all tables in this database, show the table name, column name, column type, if its optional. Also show Foreign key references to other columns. Do not show examples. Output only as json."
        response = self.agent_executor.invoke({"input": prompt, "db_name": self.db})
        self.jsonToGraph(response)
        return "Database schema visualization has been generated."

    def jsonToGraph(self, response):
        output_ = response['output']
        print("Agent:\n" + output_)
        self.parseJson(output_)

    def parseJson(self, output_):
        j = output_[output_.find('\n') + 1:output_.rfind('\n')]
        data = json.loads(j)

        G = nx.Graph()
        nodeIds = 0
        columnIds = len(data) + 1
        labeldict = {}
        canonicalColumns = dict()
        for table in data:
            nodeIds += 1
            G.add_node(nodeIds)
            G.nodes[nodeIds]['tableName'] = table["tableName"]
            labeldict[nodeIds] = table["tableName"]
            for column in table["columns"]:
                columnIds += 1
                G.add_node(columnIds)
                G.nodes[columnIds]['columnName'] = column["columnName"]
                G.nodes[columnIds]['columnType'] = column["columnType"]
                G.nodes[columnIds]['isOptional'] = column["isOptional"]
                labeldict[columnIds] = column["columnName"]
                canonicalColumns[table["tableName"] + column["columnName"]] = columnIds
                G.add_edge(nodeIds, columnIds)

        for table in data:
            for column in table["columns"]:
                if column["foreignKeyReference"] is not None:
                    this_column = table["tableName"] + column["columnName"]
                    reference_column_ = column["foreignKeyReference"]["table"] + column["foreignKeyReference"]["column"]
                    G.add_edge(canonicalColumns[this_column], canonicalColumns[reference_column_])

        print(G.number_of_nodes())
        pos = graphviz_layout(G, prog='neato')
        nx.draw(G, pos, labels=labeldict, with_labels=True)
        plt.show()

    def answer_question(self, question: str) -> str:
        """
        Answers questions about the database using the agent executor
        """
        response = self.agent_executor.invoke({"input": question, "db_name": self.db})
        return response['output']

if __name__ == "__main__":
    agent = DiscoveryAgent()
    response = agent.agent_executor.invoke({"input": "Analyse the schema of the database and its tables.", "db_name": agent.db})
    print(response['output'])