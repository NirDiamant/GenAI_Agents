from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from db_agent import DBAgent
import logging
from datetime import datetime
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

class ConversationState(TypedDict):
    question: str
    plan: Optional[List[str]]
    db_results: str
    response: str
    context: dict

class SupervisorAgent:
    def __init__(self):
        self.db_agent = DBAgent()
        self.llm = ChatOpenAI(temperature=0)

        self.planner_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a planning agent that analyzes questions and creates plans for answering them.
            The current available agents are:
            - Database Agent: Can query and understand database structure and content

            Create a plan as a list of steps. For complex questions that require multiple queries,
            break them down into sub-steps.

            Previous context: {context}
            """),
            ("user", "{question}"),
        ])

        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a response coordinator that creates final responses based on:
            Original Question: {question}
            Database Results: {db_results}
            Previous Context: {context}

            Create a clear, helpful response that directly answers the question.""")
        ])

    def create_plan(self, state: ConversationState) -> ConversationState:
        """Analyze the question and create a plan"""
        logger.info(f"Creating plan for question: {state['question']}")
        planner_response = self.llm.invoke(
            self.planner_prompt.format(
                question=state['question'],
                context=state.get('context', {})
            )
        )

        plan = [step.strip() for step in planner_response.content.split('\n') if step.strip()]
        logger.info(f"Generated plan: {plan}")

        return {
            **state,
            "plan": plan
        }

    def execute_db_steps(self, state: ConversationState) -> ConversationState:
        """Execute any database-related steps in the plan"""
        db_results = []

        for step in state['plan']:
            if any(db_term in step.lower() for db_term in ['database', 'query', 'schema']):
                logger.info(f"Delegating to DBAgent with step: {step}")
                logger.info(f"Current conversation context: {state.get('context', {})}")

                result = self.db_agent.query(step, state=state.get('context', {}))
                logger.info(f"Received response from DBAgent")
                db_results.append(f"Step: {step}\nResult: {result}")

        return {
            **state,
            "db_results": "\n".join(db_results)
        }

    def generate_response(self, state: ConversationState) -> ConversationState:
        """Generate the final response"""
        logger.info("Generating final response")
        response = self.llm.invoke(
            self.response_prompt.format(
                question=state['question'],
                db_results=state['db_results'],
                context=state.get('context', {})
            )
        )

        new_context = {
            **state.get('context', {}),
            'last_question': state['question'],
            'last_response': response.content
        }

        logger.info("Response generated successfully")
        logger.info(f"Updated context: {new_context}")

        return {
            **state,
            "response": response.content,
            "context": new_context
        }

def create_graph():
    supervisor = SupervisorAgent()

    # Build graph
    builder = StateGraph(ConversationState)

    # Add nodes
    builder.add_node("create_plan", supervisor.create_plan)
    builder.add_node("execute_db_steps", supervisor.execute_db_steps)
    builder.add_node("generate_response", supervisor.generate_response)

    # Add edges
    builder.add_edge(START, "create_plan")
    builder.add_edge("create_plan", "execute_db_steps")
    builder.add_edge("execute_db_steps", "generate_response")
    builder.add_edge("generate_response", END)

    return builder.compile()

# Create the graph
graph = create_graph()

def get_response(question: str, context: dict = None) -> dict:
    """
    Process a question through the supervisor agent.

    Args:
        question (str): The question to be answered
        context (dict, optional): Previous conversation context

    Returns:
        dict: Contains the complete response including database results
    """
    result = graph.invoke({
        "question": question,
        "plan": None,
        "db_results": "",
        "response": "",
        "context": context or {}
    })
    return result

if __name__ == "__main__":
    question = "How many tables does the database have?"
    result = get_response(question)
    print(f"\nQuestion: {question}")
    print(f"Plan: {result['plan']}")
    print(f"DB Results: {result['db_results']}")
    print(f"Response: {result['response']}")

