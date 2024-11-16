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
        1. Database: [query] - Use this prefix ONLY for direct database queries that retrieve data
        2. Research: [topic] - Use this prefix for database analysis, best practices, performance advice, and technical knowledge
        3. General: [action] - Use this prefix for responses about non-database questions

        Rules:
        - Database: steps must be specific data retrieval queries only
        - Research: steps handle all analysis, best practices, and technical advice
        - For non-database questions, use a single General: response
        - Keep responses focused on THIS database and database concepts only

        Examples:
        Question: "How many employees and albums are there?"
        Plan:
        Database: Count number of employees in the database
        Database: Count number of albums in the database

        Question: "What are the best practices for indexing in this database?"
        Plan:
        Research: Analyze current index usage and provide best practices
        Research: Recommend indexing improvements based on database structure

        Question: "What year did World War 2 end?"
        Plan:
        General: I'd love to help! While I can't answer questions about historical events, I'm an expert on this database. What would you like to know about its contents?

        Question: "How can I improve database performance?"
        Plan:
        Research: Analyze current database performance metrics
        Research: Provide optimization recommendations based on database structure

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

            # Extract database, research and general steps
            db_steps = [step for step in plan
                       if step.startswith('Database:') and len(step.split(':', 1)) == 2 and step.split(':', 1)[1].strip()]
            research_steps = [step for step in plan
                            if step.startswith('Research:') and len(step.split(':', 1)) == 2 and step.split(':', 1)[1].strip()]
            general_steps = [step for step in plan if step.startswith('General:')]

            # Log the plan
            if db_steps or research_steps:
                logger.info(f"Generated steps: Database={db_steps}, Research={research_steps}")
                return db_steps + research_steps + general_steps
            else:
                logger.info("No database or research steps found - treating as general query")
                return general_steps or ["General: I'd love to help! Please ask a specific question about the database."]

        except Exception as e:
            logger.error(f"Error creating plan: {str(e)}", exc_info=True)
            return ["General: Error occurred while creating plan"]