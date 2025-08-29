"""
Retrieval agent for finding relevant information in vector store.
"""

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from ..retrieval import Retriever, ContextRetriever

class RetrievalAgent:
    """Agent responsible for retrieving relevant information."""
    
    def __init__(self, retriever: Retriever, context_retriever: ContextRetriever, llm_model: str = "gpt-3.5-turbo"):
        """
        Initialize the retrieval agent.
        
        Args:
            retriever: Retriever for document search
            context_retriever: Context retriever for context preparation
            llm_model: LLM model to use
        """
        # Initialize components
        self.retriever = retriever
        self.context_retriever = context_retriever
        
        # Initialize LLM
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0)
        
        # Define tools available to this agent
        self.tools = [
            Tool(
                name="search",
                func=self._search,
                description="Search for information using a query. Input: query"
            ),
            Tool(
                name="get_context",
                func=self._get_context,
                description="Get detailed context for a query. Input: query"
            ),
            Tool(
                name="hybrid_search",
                func=self._hybrid_search,
                description="Perform a hybrid search with multiple query variations. Input: query"
            )
        ]
        
        # Create agent prompt
        self.agent_prompt = PromptTemplate.from_template(
            """You are a Retrieval Agent responsible for finding relevant information.
            Your job is to search for and retrieve the most relevant context for answering queries.
            
            Use the following tools to accomplish your tasks:
            {tools}
            
            Follow this process:
            1. Understand what information is needed
            2. Choose the appropriate search method
            3. Retrieve and organize the results
            
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
    
    def _search(self, query: str) -> str:
        """Search for information using a query."""
        try:
            results = self.retriever.retrieve(query, top_k=5, use_hybrid=False)
            
            # Format results
            result_text = f"Found {len(results)} results for query: '{query}'\n\n"
            
            for i, doc in enumerate(results):
                result_text += f"Result {i+1}:\n"
                result_text += f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                result_text += f"Content: {doc.page_content[:200]}...\n\n"
                
            return result_text
        except Exception as e:
            return f"Error performing search: {str(e)}"
    
    def _get_context(self, query: str) -> str:
        """Get detailed context for a query."""
        try:
            context = self.context_retriever.get_context(query, top_k=5)
            
            return f"Retrieved context for query: '{query}'\n\n{context['context'][:1000]}..."
        except Exception as e:
            return f"Error retrieving context: {str(e)}"
    
    def _hybrid_search(self, query: str) -> str:
        """Perform a hybrid search with multiple query variations."""
        try:
            results = self.retriever.retrieve(query, top_k=5, use_hybrid=True)
            
            # Format results
            result_text = f"Found {len(results)} results for hybrid search: '{query}'\n\n"
            
            for i, doc in enumerate(results):
                result_text += f"Result {i+1}:\n"
                result_text += f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                result_text += f"Query Type: {doc.metadata.get('query_type', 'Unknown')}\n"
                result_text += f"Score: {doc.metadata.get('similarity_score', 'Unknown')}\n"
                result_text += f"Content: {doc.page_content[:200]}...\n\n"
                
            return result_text
        except Exception as e:
            return f"Error performing hybrid search: {str(e)}"
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the retrieval agent with a query."""
        response = self.agent_executor.invoke({"input": query})
        
        # Try to get structured context if possible
        try:
            context = self.context_retriever.get_context(query)
            response["context"] = context
        except Exception:
            pass
            
        return response
