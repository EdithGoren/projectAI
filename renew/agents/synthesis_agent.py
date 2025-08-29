"""
Synthesis agent for creating coherent responses from retrieved information.
"""

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from langchain.schema import Document
from ..generation import ResponseGenerator, CitationManager

class SynthesisAgent:
    """Agent responsible for synthesizing information into coherent responses."""
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        """
        Initialize the synthesis agent.
        
        Args:
            llm_model: LLM model to use
        """
        # Initialize components
        self.response_generator = ResponseGenerator(model_name=llm_model)
        self.citation_manager = CitationManager()
        
        # Initialize LLM
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0.3)
        
        # Define tools available to this agent
        self.tools = [
            Tool(
                name="generate_response",
                func=self._generate_response,
                description="Generate a response from context documents. Input: JSON string with 'query' and 'documents'"
            ),
            Tool(
                name="identify_gaps",
                func=self._identify_gaps,
                description="Identify gaps in the available information. Input: JSON string with 'query' and 'context'"
            ),
            Tool(
                name="format_citations",
                func=self._format_citations,
                description="Format citations for a response. Input: JSON string with 'text' and 'format_type'"
            )
        ]
        
        # Create agent prompt
        self.agent_prompt = PromptTemplate.from_template(
            """You are a Synthesis Agent responsible for creating coherent responses from retrieved information.
            Your job is to synthesize information and generate well-cited answers to research questions.
            
            Use the following tools to accomplish your tasks:
            {tools}
            
            Follow this process:
            1. Review the available context information
            2. Identify the key points and any gaps
            3. Generate a comprehensive response with proper citations
            
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
    
    def _generate_response(self, input_str: str) -> str:
        """Generate a response from context documents."""
        try:
            # Parse input
            import json
            input_data = json.loads(input_str)
            query = input_data["query"]
            documents = input_data.get("documents", [])
            
            # Convert documents if needed
            if documents and not isinstance(documents[0], Document):
                documents = [Document(page_content=doc["content"], metadata=doc.get("metadata", {})) 
                            for doc in documents]
            
            # Generate response
            response = self.response_generator.generate_response(query, documents)
            
            return f"Generated response for query: '{query}'\n\n{response['response']}"
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _identify_gaps(self, input_str: str) -> str:
        """Identify gaps in the available information."""
        try:
            # Parse input
            import json
            input_data = json.loads(input_str)
            query = input_data["query"]
            context = input_data["context"]
            
            prompt = f"""Identify any gaps or missing information in the context provided below for answering this query:
            
            Query: {query}
            
            Context:
            {context[:2000]}...
            
            What important information is missing? What additional information would be helpful to fully answer the query?
            """
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error identifying gaps: {str(e)}"
    
    def _format_citations(self, input_str: str) -> str:
        """Format citations for a response."""
        try:
            # Parse input
            import json
            input_data = json.loads(input_str)
            text = input_data["text"]
            format_type = input_data.get("format_type", "academic")
            
            # Extract citations from text
            citations = self.citation_manager.find_citations_in_text(text)
            
            # Format citations
            formatted_citations = []
            for citation in citations:
                formatted = self.citation_manager.format_citation(citation, format_type)
                formatted_citations.append(formatted)
                
            return f"Found {len(formatted_citations)} citations:\n\n" + "\n".join(formatted_citations)
        except Exception as e:
            return f"Error formatting citations: {str(e)}"
    
    def run(self, query: str, context_docs: List[Document]) -> Dict[str, Any]:
        """Run the synthesis agent with a query and context documents."""
        # Add documents to citation manager
        self.citation_manager.add_citations(context_docs)
        
        # Create input for the agent
        import json
        docs_data = [{"content": doc.page_content, "metadata": doc.metadata} for doc in context_docs]
        input_data = {
            "query": query,
            "context": "\n\n".join(doc.page_content for doc in context_docs[:3]),
            "documents": docs_data
        }
        
        # Format as a request for the agent
        agent_input = f"""
        Research Question: {query}
        
        Number of context documents available: {len(context_docs)}
        
        Task: Generate a comprehensive response to the research question using the available context documents.
        """
        
        # Run the agent
        response = self.agent_executor.invoke({"input": agent_input})
        
        # Generate a direct response as fallback or supplement
        direct_response = self.response_generator.generate_response(query, context_docs)
        response["direct_response"] = direct_response
        
        # Add bibliography
        response["bibliography"] = self.citation_manager.get_bibliography()
        
        return response
