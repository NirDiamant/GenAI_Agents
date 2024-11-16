from typing import Annotated, TypedDict
from typing_extensions import NotRequired
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from db_agent import DBAgent
from research_agent import ResearchAgent
from planner_agent import PlannerAgent
import logging
from datetime import datetime
import httpx
from langchain.callbacks.base import BaseCallbackHandler
import operator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

def merge_context(old_context: dict, new_context: dict) -> dict:
    """Merge two context dictionaries, combining their histories."""
    # Get existing history from both contexts
    old_history = old_context.get("history", [])
    new_history = new_context.get("history", [])

    # Only add new history if it's different from the last entry
    if not old_history or old_history[-1] != new_history[-1]:
        combined_history = old_history + new_history
    else:
        combined_history = old_history

    return {
        "history": combined_history,
        "last_question": new_context.get("last_question", old_context.get("last_question", "")),
        "last_response": new_context.get("last_response", old_context.get("last_response", ""))
    }

def context_reducer(current: dict, update: dict) -> dict:
    return merge_context(current, update)

class ConversationState(TypedDict):
    question: str
    plan: NotRequired[str]
    db_results: NotRequired[str]
    response: NotRequired[str]
    context: Annotated[dict, merge_context]

class SupervisorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.db_agent = DBAgent()
        self.research_agent = ResearchAgent()
        self.planner_agent = PlannerAgent()

        # Update the response prompt to handle multiple results
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a response coordinator that creates final responses based on:
            Original Question: {question}
            Database Results: {db_results}
            Previous Context: {context}

            Rules:
            1. ALWAYS include ALL results from database queries in your response
            2. If a research question cannot be answered, acknowledge it but don't let it overshadow the database results
            3. Format the response clearly with each piece of information on its own line
            4. Use bullet points or numbers for multiple pieces of information

            Example format:
            Here are the results:
            1. [First database result]
            2. [Second database result]
            3. [Research result or acknowledgment of missing information]
            """)
        ])

    def create_plan(self, state: ConversationState) -> ConversationState:
        """Delegate plan creation to PlannerAgent"""
        plan = self.planner_agent.create_plan(
            question=state['question'],
            context=state.get('context', {})
        )

        # If plan only contains General actions, don't show any plan
        if all(step.startswith('General:') for step in plan):
            new_state = {
                **state,
                "plan": plan  # Keep the plan for processing but don't display it
            }
            return new_state

        # Filter to show only Database actions in the plan preview
        db_steps = [step for step in plan if step.startswith('Database:')]

        # Only ask for confirmation if there are multiple database steps
        if len(db_steps) > 3:
            print("\nHere's my planned approach:")
            for step in db_steps:  # Only show database steps
                print(f"  {step}")

            confirmation = input("\nWould you like me to proceed with this plan? (yes/no): ").lower().strip()

            if confirmation != 'yes':
                new_state = {
                    **state,
                    "plan": None,
                    "response": "No problems, how can I help you next?"
                }
                logger.info(f"Plan rejected.")
                return new_state

        new_state = {
            **state,
            "plan": plan  # Keep the full plan including General actions
        }
        logger.info(f"Plan created.")
        return new_state

    def execute_plan(self, state: ConversationState) -> ConversationState:
        """Execute database, research, and general steps in the plan"""
        db_results = []
        research_results = []
        general_results = []

        try:
            for step in state['plan']:
                # Extract the step type and content
                if not (':' in step):
                    continue

                step_type, content = step.split(':', 1)
                step_type = step_type.lower().strip()
                content = content.strip()

                if step_type == 'research':
                    logger.info(f"Delegating to ResearchAgent: {content}")
                    try:
                        result = self.research_agent.research(content, state=state.get('context', {}))
                        logger.info("Research step completed successfully")
                        research_results.append(f"Step: {step}\nResult: {result}")
                    except Exception as e:
                        logger.error(f"Error in research step: {str(e)}", exc_info=True)
                        research_results.append(f"Step: {step}\nError: Research failed - {str(e)}")

                elif step_type == 'database':
                    logger.info(f"Delegating to DBAgent: {content}")
                    try:
                        result = self.db_agent.query(content, state=state.get('context', {}))
                        db_results.append(f"Step: {step}\nResult: {result}")
                    except Exception as e:
                        logger.error(f"Error in database step: {str(e)}", exc_info=True)
                        db_results.append(f"Step: {step}\nError: Database query failed - {str(e)}")

                elif step_type == 'general':
                    logger.info(f"Handling general action: {content}")
                    general_results.append(f"Step: {step}\nResult: {content}")

            all_results = db_results + research_results + general_results

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
        """Generate the final response"""
        # logger.info(f"Current state: {state}")

        # If plan was rejected, return early with the stored response
        if state.get("plan") is None and "response" in state:
            logger.info(f"Using existing response. State unchanged: {state}")
            return state

        logger.info("Generating final response")
        response = self.llm.invoke(
            self.response_prompt.format(
                question=state['question'],
                db_results=state.get('db_results', ''),
                context=state.get('context', {})
            )
        )

        # Only update the context if it hasn't been updated by a delegated agent
        if not state.get('context', {}).get('history', []) or \
           state['question'] != state['context'].get('last_question', ''):
            new_context = {
                'history': state.get('context', {}).get('history', []) + [{
                    'question': state['question'],
                    'response': response.content
                }],
                'last_question': state['question'],
                'last_response': response.content
            }
        else:
            new_context = state.get('context', {})

        new_state = {
            **state,
            "response": response.content,
            "context": new_context
        }
        logger.info(f"Response generated.")
        return new_state

def create_graph():
    supervisor = SupervisorAgent()

    # Build graph
    builder = StateGraph(ConversationState)

    # Add nodes
    builder.add_node("create_plan", supervisor.create_plan)
    builder.add_node("execute_plan", supervisor.execute_plan)
    builder.add_node("generate_response", supervisor.generate_response)

    # Add edges with conditional routing
    builder.add_edge(START, "create_plan")

    # Only go to execute_plan if we have a plan
    builder.add_conditional_edges(
        "create_plan",
        lambda x: "execute_plan" if x.get("plan") is not None else "generate_response"
    )

    builder.add_edge("execute_plan", "generate_response")
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


