from typing import Annotated, TypedDict
from typing_extensions import NotRequired
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from inference_agent import InferenceAgent
from planner_agent import PlannerAgent
import logging
import operator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def merge_context(old_context: dict, new_context: dict) -> dict:
    """Merge two context dictionaries, combining their histories."""
    old_history = old_context.get("history", [])
    new_history = new_context.get("history", [])

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
        self.inference_agent = InferenceAgent()
        self.planner_agent = PlannerAgent()

        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a response coordinator that creates final responses based on:
            Original Question: {question}
            Database Results: {db_results}
            Previous Context: {context}

            Rules:
            1. ALWAYS include ALL results from database queries in your response
            2. Format the response clearly with each piece of information on its own line
            3. Use bullet points or numbers for multiple pieces of information
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

        # Filter to show only Inference actions in the plan preview
        inference_steps = [step for step in plan if step.startswith('Inference:')]

        # Only ask for confirmation if there are multiple steps
        if len(inference_steps) > 3:
            print("\nHere's my planned approach:")
            for step in inference_steps:
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
            "plan": plan
        }
        logger.info(f"Plan created.")
        return new_state

    def execute_plan(self, state: ConversationState) -> ConversationState:
        """Execute inference and general steps in the plan"""
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
                        result = self.inference_agent.query(content, state=state.get('context', {}))
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
        """Generate the final response"""
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


