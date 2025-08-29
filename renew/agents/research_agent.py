"""
Research agent for analyzing questions and planning research strategies.
"""

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from ..retrieval import QueryProcessor

class ResearchAgent:
    """Agent responsible for research planning and query formulation."""
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        """
        Initialize the research agent.
        
        Args:
            llm_model: LLM model to use
        """
        # Initialize components
        self.query_processor = QueryProcessor()
        
        # Initialize LLM
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0.2)
        
        # Define tools available to this agent
        self.tools = [
            Tool(
                name="analyze_question",
                func=self._analyze_question,
                description="Analyze a research question and break it down into components. Input: question"
            ),
            Tool(
                name="formulate_queries",
                func=self._formulate_queries,
                description="Generate optimized search queries for a research question. Input: question"
            ),
            Tool(
                name="expand_query",
                func=self._expand_query,
                description="Expand a query with academic terminology. Input: query"
            )
        ]
        
        # Create agent prompt
        self.agent_prompt = PromptTemplate.from_template(
            """You are a Research Planning Agent responsible for analyzing research questions and planning search strategies.
            Your job is to break down complex questions into searchable components.
            
            Use the following tools to accomplish your tasks:
            {tools}
            
            Follow this process:
            1. Analyze the research question in detail
            2. Break it into searchable components
            3. Formulate optimized queries for each component
            
            Use the format:
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
            
            Question: {input}
            {agent_scratchpad}
            """
        )
        
        # Create the agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.agent_prompt
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True  # Add this parameter to handle parsing errors
        )
    
    def _analyze_question(self, question: str) -> str:
        """Analyze a research question and break it down."""
        prompt = f"""Analyze the following research question and break it down into components:
        
        Question: {question}
        
        1. Main topic:
        2. Key concepts:
        3. Subtopics:
        4. Potential research angles:
        5. Required background knowledge:
        
        Provide your analysis in the format above.
        """
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def _formulate_queries(self, question: str) -> str:
        """Generate optimized search queries for a question."""
        prompt = f"""Generate 3-5 optimized search queries for the following research question:
        
        Question: {question}
        
        Each query should:
        - Focus on a specific aspect of the question
        - Use precise academic terminology
        - Be clear and concise
        
        Format each query on a new line.
        """
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def _expand_query(self, query: str) -> str:
        """Expand a query with academic terminology."""
        try:
            expanded = self.query_processor.expand_query(query)
            return f"Original query: {query}\nExpanded query: {expanded}"
        except Exception as e:
            return f"Error expanding query: {str(e)}"
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the research agent with a query."""
        response = self.agent_executor.invoke({"input": query})
        
        # Extract the search queries from the response
        # This would need to be parsed from the agent's output
        # For now, just pass through the response
        return response
