from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from db_agent import DBAgent

class State(TypedDict):
    question: str
    db_results: str
    response: str

class SupervisorAgent:
    def __init__(self):
        self.db_agent = DBAgent()

        self.tools = [
            Tool(
                name="query_database",
                func=self._consult_db_agent,
                description="Consult the database agent for information"
            ),
            Tool(
                name="generate_response",
                func=lambda x: self._generate_response(x['question'], x['db_results']),
                description="Generate a final response based on database results. Requires both question and db_results."
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a supervisor agent responsible for handling user questions about data.
            Always follow these steps:
            1. First consult the database agent for relevant information
            2. Store the database results
            3. Generate a response using both the question and database results

            Format your final response as:
            Database Results: [query results]
            Response: [your explanation of the results]"""),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])

        self.agent = create_openai_functions_agent(
            llm=ChatOpenAI(temperature=0),
            prompt=prompt,
            tools=self.tools
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )

    def _consult_db_agent(self, text: str) -> str:
        """Consult the database agent for information"""
        return self.db_agent.query(text)

    def _generate_response(self, question: str, db_results: str) -> str:
        """Generate a response based on database results"""
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        return llm.invoke(
            f"""Based on these database results: {db_results}
            Generate a helpful response to: {question}
            Explain the findings in a clear way."""
        ).content

    def process_request(self, state: State) -> State:
        """Process a request through the supervisor agent"""
        result = self.executor.invoke({
            "input": f"Handle this database question: {state['question']}"
        })

        return {
            **state,
            "response": result["output"]
        }

# Initialize the graph
def create_graph():
    supervisor = SupervisorAgent()

    # Build graph
    builder = StateGraph(State)

    # Add the main processing node
    builder.add_node("process_request", supervisor.process_request)

    # Add edges
    builder.add_edge(START, "process_request")
    builder.add_edge("process_request", END)

    return builder.compile()

# Create the graph
graph = create_graph()

def get_response(question: str) -> dict:
    """
    Process a question through the supervisor agent.

    Args:
        question (str): The question to be answered

    Returns:
        dict: Contains the complete response including database results
    """
    result = graph.invoke({
        "question": question,
        "db_results": "",
        "response": ""
    })
    return result

if __name__ == "__main__":
    question = "How many tables does the database have?"
    result = get_response(question)
    print(f"\nQuestion: {question}")
    print(f"Result: {result['response']}")

