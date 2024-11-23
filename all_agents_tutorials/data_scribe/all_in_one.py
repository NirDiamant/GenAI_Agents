import json
import logging
import os
import sys
from typing import Annotated, TypedDict, List, Optional
from typing_extensions import NotRequired

import matplotlib.pyplot as plt
import networkx as nx
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_agraph import graphviz_layout
import pickle

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Disable HTTP request logging
class Config:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.db = os.getenv("DATABASE")
        self.graph_cache_path = "discover.pkl"
        
        if not all([self.openai_api_key, self.db]):
            raise ValueError("Missing required environment variables: OPENAI_API_KEY, DATABASE")
        
        self.db_engine = SQLDatabase.from_uri(f"sqlite:///{self.db}")
        self.llm = ChatOpenAI(temperature=0)
        self.llm_gpt4 = ChatOpenAI(temperature=0, model_name="gpt-4")

class DiscoveryAgent:
    def __init__(self):
        self.config = Config()
        self.toolkit = SQLDatabaseToolkit(db=self.config.db_engine, llm=self.config.llm_gpt4)
        self.tools = self.toolkit.get_tools()

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

        self.chat_prompt = self.create_chat_prompt()
        self.agent = create_openai_functions_agent(
            llm=self.config.llm_gpt4,
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

    def run_query(self, q):
        return self.config.db_engine.run(q)

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
                            "columnName": [COLUMN 1 NAME],
                            "columnType": [COLUMN 1 TYPE],
                            "isOptional": [true OR false],
                            "foreignKeyReference": {{
                                "table": [REFERENCE TABLE NAME],
                                "column": [REFERENCE COLUMN NAME]
                            }}
                        }},
                        {{
                            "columnName": [COLUMN 2 NAME],
                            "columnType": [COLUMN 2 TYPE],
                            "isOptional": [true OR false],
                            "foreignKeyReference": {{
                                "table": [REFERENCE TABLE NAME],
                                "column": [REFERENCE COLUMN NAME]
                            }}
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

    def save_graph_db(self, graph: nx.Graph) -> None:
        """Save the NetworkX graph to disk"""
        try:
            with open(self.config.graph_cache_path, 'wb') as f:
                pickle.dump(graph, f)
            logger.info(f"Graph saved to {self.config.graph_cache_path}")
        except Exception as e:
            logger.error(f"Failed to save graph: {str(e)}")

    def load_graph_db(self) -> Optional[nx.Graph]:
        """Load the NetworkX graph from disk if it exists"""
        try:
            if os.path.exists(self.config.graph_cache_path):
                with open(self.config.graph_cache_path, 'rb') as f:
                    graph = pickle.load(f)
                logger.info(f"Graph loaded from {self.config.graph_cache_path}")
                return graph
        except Exception as e:
            logger.error(f"Failed to load graph: {str(e)}")
        return None

    def discover(self) -> nx.Graph:
        """Modified to check for cached graph first"""
        cached_graph = self.load_graph_db()
        if cached_graph is not None:
            return cached_graph

        logger.info("No cached graph found, performing discovery...")
        prompt = "For all tables in this database, show the table name, column name, column type, if its optional. Also show Foreign key references to other columns. Do not show examples. Output only as json."
        response = self.agent_executor.invoke({"input": prompt, "db_name": self.config.db})
        
        graph = self.jsonToGraph(response)
        self.save_graph_db(graph)
        return graph

    def jsonToGraph(self, response):
        output_ = response['output']
        return self.parseJson(output_)

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

        return G

    def answer_question(self, question: str) -> str:
        response = self.agent_executor.invoke({"input": question, "db_name": self.config.db})
        return response['output']

class InferenceAgent:
    def __init__(self):
        self.config = Config()
        self.toolkit = SQLDatabaseToolkit(db=self.config.db_engine, llm=self.config.llm)
        self.tools = self.toolkit.get_tools()
        self.chat_prompt = self.create_chat_prompt()
        
        self.agent = create_openai_functions_agent(
            llm=self.config.llm,
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
        try:
            self.show_tables()
            logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise

    def show_tables(self) -> str:
        q = '''
            SELECT
                name,
                type
            FROM sqlite_master
            WHERE type IN ("table","view");
            '''
        return self.run_query(q)

    def run_query(self, q: str) -> str:
        try:
            return self.config.db_engine.run(q)
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return f"Error executing query: {str(e)}"

    def create_chat_prompt(self) -> ChatPromptTemplate:
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

    def analyze_question_with_graph(self, db_graph: nx.Graph, question: str) -> dict:
        """Use the graph structure to understand how to answer the question"""
        print(f"\n🔎 Starting graph analysis for: '{question}'")

        # Convert question to lowercase for matching
        question_lower = question.lower()

        analysis = {
            'tables': [],
            'relationships': [],
            'columns': [],
            'possible_paths': []
        }

        print("\n📋 Scanning graph nodes for relevant tables and columns...")
        for node in db_graph.nodes():
            node_data = db_graph.nodes[node]

            # Check if it's a table node
            if 'tableName' in node_data:
                table_name = node_data['tableName'].lower()
                # Only include table if it or its common variations appear in the question
                if (table_name in question_lower or
                    table_name.rstrip('s') in question_lower or  # singular form
                    f"{table_name}s" in question_lower):        # plural form

                    print(f"  📦 Found relevant table: {node_data['tableName']}")
                    columns = []
                    for neighbor in db_graph.neighbors(node):
                        col_data = db_graph.nodes[neighbor]
                        if 'columnName' in col_data:
                            col_name = col_data['columnName'].lower()
                            # Only include column if it appears in the question
                            if col_name in question_lower:
                                columns.append({
                                    'name': col_data['columnName'],
                                    'type': col_data['columnType'],
                                    'table': node_data['tableName']
                                })
                                print(f"    📎 Found relevant column: {col_data['columnName']}")

                    analysis['tables'].append({
                        'name': node_data['tableName'],
                        'columns': columns
                    })

        # Only look for paths between relevant tables
        if len(analysis['tables']) > 1:
            print("\n🔗 Finding relationships between relevant tables...")
            table_nodes = [n for n in db_graph.nodes()
                          if db_graph.nodes[n].get('tableName') in [t['name'] for t in analysis['tables']]]

            for i, start_node in enumerate(table_nodes):
                for end_node in table_nodes[i+1:]:
                    try:
                        path = nx.shortest_path(db_graph, start_node, end_node)
                        named_path = []
                        for node in path:
                            node_data = db_graph.nodes[node]
                            if 'tableName' in node_data:
                                named_path.append(f"Table: {node_data['tableName']}")
                            elif 'columnName' in node_data:
                                named_path.append(f"Column: {node_data['columnName']}")
                        analysis['possible_paths'].append(named_path)
                        print(f"  ↔️  Found path: {' -> '.join(named_path)}")
                    except nx.NetworkXNoPath:
                        continue

        print("\n✅ Graph analysis complete")
        return analysis

    def query(self, text: str, db_graph) -> str:
        try:
            if db_graph:
                print(f"\n🔍 Analyzing query with graph: '{text}'")
                # Analyze the question using the graph structure
                graph_analysis = self.analyze_question_with_graph(db_graph, text)
                print(f"\n📊 Graph Analysis Results:")
                print(json.dumps(graph_analysis, indent=2))

                # Add the graph analysis to the context for the LLM
                enhanced_prompt = f"""
                Database Structure Analysis:
                - Available Tables: {[t['name'] for t in graph_analysis['tables']]}
                - Table Relationships: {graph_analysis['possible_paths']}

                User Question: {text}

                Use this structural information to form an accurate query.
                """
                print(f"\n📝 Enhanced prompt created with graph context")
                return self.agent_executor.invoke({"input": enhanced_prompt, "db_name": self.config.db})['output']

            print(f"\n⚡ No graph available, executing standard query: '{text}'")
            return self.agent_executor.invoke({"input": text, "db_name": self.config.db})['output']

        except Exception as e:
            print(f"\n❌ Error in inference query: {str(e)}")
            return f"Error processing query: {str(e)}"

class PlannerAgent:
    def __init__(self):
        self.config = Config()
        self.planner_prompt = self.create_planner_prompt()

    def create_planner_prompt(self):
        system_template = """You are a friendly planning agent that creates specific plans to answer questions about THIS database only.

        Available actions:
        1. Inference: [query] - Use this prefix for database queries
        2. General: [response] - Use this prefix for friendly responses

        Create a SINGLE, SEQUENTIAL plan where:
        - Each step should be exactly ONE line
        - Each step must start with either 'Inference:' or 'General:'
        - Steps must be in logical order
        - DO NOT repeat steps
        - Keep the plan minimal and focused

        Example format:
        Inference: Get all artists from the database
        Inference: Count tracks per artist
        General: Provide the results in a friendly way
        """

        human_template = "Question: {question}\n\nCreate a focused plan with appropriate action steps."

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    def create_plan(self, question: str, context: dict = None) -> list:
        try:
            logger.info(f"Creating plan for question: {question}")
            planner_response = self.llm.invoke(
                self.planner_prompt.format(
                    question=question,
                    context=context or {}
                )
            )
            # Get all steps, removing empty lines and 'plan:' header
            plan = [step.strip() for step in planner_response.content.split('\n')
                   if step.strip() and not step.lower() == 'plan:']

            inference_steps = [step for step in plan
                           if step.startswith('Inference:') and len(step.split(':', 1)) == 2 and step.split(':', 1)[1].strip()]
            general_steps = [step for step in plan if step.startswith('General:')]

            if inference_steps or general_steps:
                logger.info(f"Generated steps: Inference={inference_steps}, General={general_steps}")
                return inference_steps + general_steps
            elif general_steps:
                logger.info("Conversational response only")
                return general_steps
            else:
                logger.info("No valid steps found - providing friendly default")
                return ["General: I'd love to help you explore the database! What would you like to know?"]

        except Exception as e:
            logger.error(f"Error creating plan: {str(e)}", exc_info=True)
            return ["General: Error occurred while creating plan"]

# Define reducers
def db_graph_reducer():
    def _reducer(previous_value: Optional[nx.Graph], new_value: nx.Graph) -> nx.Graph:
        if previous_value is None:
            return new_value
        return previous_value
    return _reducer

def plan_reducer():
    def _reducer(previous_value: Optional[List[str]], new_value: List[str]) -> List[str]:
        return new_value if new_value is not None else previous_value
    return _reducer

def classify_input_reducer():
    def _reducer(previous_value: Optional[str], new_value: str) -> str:
        return new_value  # Always use the latest classification
    return _reducer

class ConversationState(TypedDict):
    question: str
    input_type: Annotated[str, classify_input_reducer()]  # Add classification field
    plan: Annotated[List[str], plan_reducer()]
    db_results: NotRequired[str]
    response: NotRequired[str]
    db_graph: Annotated[Optional[nx.Graph], db_graph_reducer()] = None

def classify_user_input(state: ConversationState) -> ConversationState:
    """Classifies user input to determine if it requires database access."""

    system_prompt = """You are an input classifier. Classify the user's input into one of these categories:
    - DATABASE_QUERY: Questions about data, requiring database access
    - GREETING: General greetings, how are you, etc.
    - CHITCHAT: General conversation not requiring database
    - FAREWELL: Goodbye messages

    Respond with ONLY the category name."""

    messages = [
        ("system", system_prompt),
        ("user", state['question'])
    ]

    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(messages)
    classification = response.content.strip()

    logger.info(f"Input classified as: {classification}")

    return {
        **state,
        "input_type": classification
    }

class SupervisorAgent:
    def __init__(self):
        self.config = Config()
        self.inference_agent = InferenceAgent()
        self.planner_agent = PlannerAgent()
        self.discovery_agent = DiscoveryAgent()

        # Separate prompts for different types of responses
        self.db_response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a response coordinator that creates final responses based on:
            Original Question: {question}
            Database Results: {db_results}

            Rules:
            1. ALWAYS include ALL results from database queries in your response
            2. Format the response clearly with each piece of information on its own line
            3. Use bullet points or numbers for multiple pieces of information
            """)
        ])

        self.chat_response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly AI assistant.
            Respond naturally to the user's message.
            Keep responses brief and friendly.
            Don't make up information about weather, traffic, or other external data.
            """)
        ])

    def create_plan(self, state: ConversationState) -> ConversationState:
        plan = self.planner_agent.create_plan(
            question=state['question']
        )

        # Format the plan steps more readably
        logger.info("Generated plan:")
        inference_steps = [step for step in plan if step.startswith('Inference:')]
        general_steps = [step for step in plan if step.startswith('General:')]

        if inference_steps:
            logger.info("Inference Steps:")
            for i, step in enumerate(inference_steps, 1):
                logger.info(f"  {i}. {step}")
        if general_steps:
            logger.info("General Steps:")
            for i, step in enumerate(general_steps, 1):
                logger.info(f"  {i}. {step}")

        return {
            **state,
            "plan": plan
        }

    def execute_plan(self, state: ConversationState) -> ConversationState:
        inference_results = []
        general_results = []

        try:
            for step in state['plan']:
                if not (':' in step):
                    continue

                step_type, content = step.split(':', 1)
                step_type = step_type.lower().strip()
                content = content.strip()

                if step_type == 'inference':
                    logger.info(f"Delegating to InferenceAgent: {content}")
                    try:
                        result = self.inference_agent.query(content, state.get('db_graph'))
                        inference_results.append(f"Step: {step}\nResult: {result}")
                    except Exception as e:
                        logger.error(f"Error in inference step: {str(e)}", exc_info=True)
                        inference_results.append(f"Step: {step}\nError: Query failed - {str(e)}")

                elif step_type == 'general':
                    logger.info(f"Handling general action: {content}")
                    general_results.append(f"Step: {step}\nResult: {content}")

            all_results = inference_results + general_results

            if not all_results:
                logger.info("No steps were found in the plan")
                return {
                    **state,
                    "db_results": "No results were generated as no valid steps were found."
                }

            new_state = {
                **state,
                "db_results": "\n\n".join(all_results)
            }
            logger.info(f"Steps executed.")
            return new_state
        except Exception as e:
            logger.error(f"Error in execute_plan: {str(e)}", exc_info=True)
            new_state = {
                **state,
                "db_results": f"Error executing steps: {str(e)}"
            }
            logger.info(f"Execution error.")
            return new_state

    def generate_response(self, state: ConversationState) -> ConversationState:
        logger.info("Generating final response")

        if state.get("input_type") in ["GREETING", "CHITCHAT", "FAREWELL"]:
            # Use chat prompt for non-database interactions
            response = self.llm.invoke(
                self.chat_response_prompt.format(
                    question=state['question']
                )
            )
        else:
            # Use database prompt for database queries
            response = self.llm.invoke(
                self.db_response_prompt.format(
                    question=state['question'],
                    db_results=state.get('db_results', '')
                )
            )

        logger.info("Response generated.")
        return {
            **state,
            "response": response.content,
            "plan": []  # Clear the plan for the next cycle
        }

    def visualize_db_graph(self):
        """Visualize the database graph structure using matplotlib and graphviz."""
        plt.figure(figsize=(12, 8))

        # Create label dictionary from node attributes
        labeldict = {}
        for node in self.db_graph.nodes():
            # Get the 'label' attribute if it exists, otherwise use the node name
            labeldict[node] = self.db_graph.nodes[node].get('label', node)

        # Create the layout and draw the graph
        pos = graphviz_layout(self.db_graph, prog='neato')
        nx.draw(self.db_graph,
                pos,
                labels=labeldict,
                with_labels=True,
                node_color='lightblue',
                node_size=2000,
                font_size=8,
                font_weight='bold',
                edge_color='gray')

        plt.title("Database Schema Graph")
        plt.show()

def create_graph():
    supervisor = SupervisorAgent()
    builder = StateGraph(ConversationState)

    # Add all nodes
    builder.add_node("classify_input", classify_user_input)
    builder.add_node("discover_database", discover_database)
    builder.add_node("create_plan", supervisor.create_plan)
    builder.add_node("execute_plan", supervisor.execute_plan)
    builder.add_node("generate_response", supervisor.generate_response)

    # Define the flow
    builder.add_edge(START, "classify_input")

    # Only proceed to database discovery if it's a database query
    builder.add_conditional_edges(
        "classify_input",
        lambda x: "discover_database" if x.get("input_type") == "DATABASE_QUERY" else "generate_response"
    )

    builder.add_edge("discover_database", "create_plan")
    builder.add_conditional_edges(
        "create_plan",
        lambda x: "execute_plan" if x.get("plan") is not None else "generate_response"
    )
    builder.add_edge("execute_plan", "generate_response")
    builder.add_edge("generate_response", END)

    return builder.compile()

def discover_database(state: ConversationState) -> ConversationState:
    # Only discover if db_graph is None
    if state.get('db_graph') is None:
        logger.info("Performing one-time database schema discovery...")
        discovery_agent = DiscoveryAgent()
        graph = discovery_agent.discover()
        logger.info("Database schema discovery complete - this will be reused for future queries")
        return {**state, "db_graph": graph}
    return state

if __name__ == "__main__":
    graph = create_graph()

    state = graph.invoke({
        "question": "Hi there, how goes it?"
    })
    print(f"State after first invoke: {state}")
    print(f"Response 1: {state['response']}\n")

    state = graph.invoke({
        **state,
        "question": "Who are the top 3 artists by number of tracks?"
    })
    print(f"State after second invoke: {state}")
    print(f"Response 2: {state['response']}\n")

    state = graph.invoke({
        **state,
        "question": "What genres do they make?"
    })
    print(f"State after third invoke: {state}")
    print(f"Response 3: {state['response']}\n")
