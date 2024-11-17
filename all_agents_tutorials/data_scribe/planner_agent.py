import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

logger = logging.getLogger(__name__)

class PlannerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.planner_prompt = self.create_planner_prompt()

    def create_planner_prompt(self):
        system_template = """You are a friendly planning agent that creates specific plans to answer questions about THIS database only.

        Available actions:
        1. Inference: [query/analysis] - Use this prefix for database queries, analysis, insights, and best practices
        2. General: [response] - Use this prefix for actual friendly responses (not actions)

        Rules:
        - For database questions, create specific Inference: steps for each distinct query or analysis needed
        - If a question contains both database and non-database parts:
          * Create Inference: steps for the database parts
          * Add a General: with the actual response for non-database parts
        - Be friendly and conversational while staying focused on database capabilities
        - For greetings or conversation, provide the actual response, not an action

        Examples:
        Question: "Hi! Can you count employees and albums?"
        Plan:
        General: Hi there! I'd be happy to help you with that information.
        Inference: Count and analyze the number of employees
        Inference: Count and analyze the number of albums

        Question: "Count employees and tell me about World War 2"
        Plan:
        Inference: Count and analyze the number of employees
        General: While I can't provide information about World War 2, I can tell you all about the employee data in our database!

        Question: "How many employees, invoices, and tables are there?"
        Plan:
        Inference: Count and analyze the number of employees
        Inference: Calculate and analyze total invoices
        Inference: Count and analyze the number of tables

        Question: "Hi, how are you today?"
        Plan:
        General: Hello! I'm doing great, thank you for asking. I'm ready to help you explore our database. What would you like to know about?

        Previous context: {context}
        """

        human_template = "Question: {question}\n\nCreate a focused plan with appropriate action steps."

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    def create_plan(self, question: str, context: dict = None) -> list:
        """
        Create a plan for answering the user's question.
        """
        try:
            logger.info(f"Creating plan for question: {question}")

            planner_response = self.llm.invoke(
                self.planner_prompt.format(
                    question=question,
                    context=context or {}
                )
            )

            # Split into lines and clean up
            plan = [step.strip() for step in planner_response.content.split('\n')
                   if step.strip() and not step.lower() == 'plan:']

            # Extract inference and general steps
            inference_steps = [step for step in plan
                           if step.startswith('Inference:') and len(step.split(':', 1)) == 2 and step.split(':', 1)[1].strip()]
            general_steps = [step for step in plan if step.startswith('General:')]

            # Log the plan
            if inference_steps:
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