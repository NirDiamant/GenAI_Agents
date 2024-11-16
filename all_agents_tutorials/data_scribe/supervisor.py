from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from db_agent import DBAgent
from research_agent import ResearchAgent
import logging
from datetime import datetime
import httpx
from langchain.callbacks.base import BaseCallbackHandler

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
        self.research_agent = ResearchAgent()
        self.llm = ChatOpenAI(temperature=0)

        self.planner_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a planning agent that analyzes questions and creates plans for answering them.
            The current available agents are:
            - Database Agent: Can query and understand database structure and content
            - Research Agent: Can provide general knowledge and best practices about databases

            Create a plan as a list of steps. For complex questions that require multiple queries,
            break them down into sub-steps. If the question requires both practical database information
            and theoretical knowledge, use both agents.

            Example steps:
            1. Research: Find best practices for database indexing
            2. Database: Check current index usage in the database

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

        # Show plan to user and get confirmation
        print("\nHere's my planned approach:")
        for step in plan:
            print(f"  {step}")

        confirmation = input("\nWould you like me to proceed with this plan? (yes/no): ").lower().strip()

        if confirmation != 'yes':
            return {
                **state,
                "plan": None,
                "response": "No problems, how can I help you next?"
            }

        return {
            **state,
            "plan": plan
        }

    def execute_db_steps(self, state: ConversationState) -> ConversationState:
        """Execute database and research steps in the plan"""
        db_results = []
        research_results = []

        try:
            for step in state['plan']:
                step_lower = step.lower()
                # Remove the step number from the beginning (e.g., "1. Research:" becomes "Research:")
                step_content = step_lower.split('. ', 1)[-1] if '. ' in step_lower else step_lower

                # Check for research steps
                if 'research' in step_content[:10]:  # Check the beginning of the step content
                    logger.info(f"Delegating to ResearchAgent with step: {step}")
                    try:
                        result = self.research_agent.research(step, state=state.get('context', {}))
                        logger.info("Research step completed successfully")
                        research_results.append(f"Step: {step}\nResult: {result}")
                    except Exception as e:
                        logger.error(f"Error in research step: {str(e)}", exc_info=True)
                        research_results.append(f"Step: {step}\nError: Research failed - {str(e)}")

                # Check for database steps
                elif any(db_term in step_content for db_term in ['database:', 'query:', 'schema:']):
                    logger.info(f"Delegating to DBAgent with step: {step}")
                    try:
                        result = self.db_agent.query(step, state=state.get('context', {}))
                        db_results.append(f"Step: {step}\nResult: {result}")
                    except Exception as e:
                        logger.error(f"Error in database step: {str(e)}", exc_info=True)
                        db_results.append(f"Step: {step}\nError: Database query failed - {str(e)}")
                else:
                    logger.warning(f"Step type not recognized: {step}")

            all_results = db_results + research_results

            if not all_results:
                logger.warning("No results were generated from any steps")
                return {
                    **state,
                    "db_results": "No results were generated from the execution steps."
                }

            return {
                **state,
                "db_results": "\n\n".join(all_results)
            }
        except Exception as e:
            logger.error(f"Error in execute_db_steps: {str(e)}", exc_info=True)
            return {
                **state,
                "db_results": f"Error executing steps: {str(e)}"
            }

    def generate_response(self, state: ConversationState) -> ConversationState:
        """Generate the final response"""
        # If plan was rejected, return early with the stored response
        if state.get("plan") is None and "response" in state:
            return state

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

