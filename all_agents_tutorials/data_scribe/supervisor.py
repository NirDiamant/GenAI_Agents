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
            ("system", """You are a planning agent that creates specific plans to answer user questions about databases.

            Available actions:
            1. Database: [query] - Use this prefix for any database queries or analysis
            2. Research: [topic] - Use this prefix for getting best practices or general knowledge

            Rules:
            - Every action step MUST start with either "Database:" or "Research:"
            - Only create steps that require database queries or research
            - Do not include general steps like "greet user" or "summarize findings"
            - If the user's question is a greeting or too general, respond with a single step:
              Research: Inform user we need a specific database-related question

            Examples:
            Question: "How many tables are in the database?"
            Plan:
            Database: Count all tables in the schema

            Question: "What are the best practices for indexing in this database?"
            Plan:
            Database: Analyze current index usage in the database
            Research: Get best practices for database indexing
            Database: Compare current indexes against best practices

            Previous context: {context}
            """),
            ("user", "Question: {question}\n\nCreate a focused plan with only necessary Database: or Research: steps."),
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
        logger.info(f"Current state: {state}")
        logger.info(f"Creating plan for question: {state['question']}")
        planner_response = self.llm.invoke(
            self.planner_prompt.format(
                question=state['question'],
                context=state.get('context', {})
            )
        )

        plan = [step.strip() for step in planner_response.content.split('\n') if step.strip()]
        logger.info(f"Generated plan: {plan}")

        # Only ask for confirmation if plan has more than 3 steps
        if len(plan) > 3:
            print("\nHere's my planned approach:")
            for step in plan:
                print(f"  {step}")

            confirmation = input("\nWould you like me to proceed with this plan? (yes/no): ").lower().strip()

            if confirmation != 'yes':
                new_state = {
                    **state,
                    "plan": None,
                    "response": "No problems, how can I help you next?"
                }
                logger.info(f"Plan rejected. New state: {new_state}")
                return new_state

        new_state = {
            **state,
            "plan": plan
        }
        logger.info(f"Plan created. New state: {new_state}")
        return new_state

    def execute_db_steps(self, state: ConversationState) -> ConversationState:
        """Execute database and research steps in the plan"""
        logger.info(f"Current state: {state}")
        db_results = []
        research_results = []

        try:
            for step in state['plan']:
                # Skip any steps that don't start with Database: or Research:
                if not (step.lower().startswith('database:') or step.lower().startswith('research:')):
                    continue

                # Extract the step type and content
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

            all_results = db_results + research_results

            if not all_results:
                logger.info("No delegatable steps were found in the plan")
                return {
                    **state,
                    "db_results": "No results were generated as no database or research steps were needed."
                }

            new_state = {
                **state,
                "db_results": "\n\n".join(all_results)
            }
            logger.info(f"Steps executed. New state: {new_state}")
            return new_state
        except Exception as e:
            logger.error(f"Error in execute_db_steps: {str(e)}", exc_info=True)
            new_state = {
                **state,
                "db_results": f"Error executing steps: {str(e)}"
            }
            logger.info(f"Execution error. New state: {new_state}")
            return new_state

    def generate_response(self, state: ConversationState) -> ConversationState:
        """Generate the final response"""
        logger.info(f"Current state: {state}")

        # If plan was rejected, return early with the stored response
        if state.get("plan") is None and "response" in state:
            logger.info(f"Using existing response. State unchanged: {state}")
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

        new_state = {
            **state,
            "response": response.content,
            "context": new_context
        }
        logger.info(f"Response generated. New state: {new_state}")
        return new_state

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

