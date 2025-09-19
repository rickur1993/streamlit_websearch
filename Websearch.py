import streamlit as st
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import os
import json
from dotenv import load_dotenv
import os
import requests
#load_dotenv("API.env")
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Try importing the new Google GenAI SDK first (recommended)
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
XAI_API_KEY = st.secrets["XAI_API_KEY"]
try:
    from google import genai
    from google.genai import types
    NEW_SDK_AVAILABLE = True
    OLD_SDK_AVAILABLE = False
except ImportError:
    NEW_SDK_AVAILABLE = False
    # Fallback to old SDK
    try:
        import google.generativeai as genai_old
        OLD_SDK_AVAILABLE = True
        try:
            from google.generativeai.types import ToolConfig
            TOOL_CONFIG_AVAILABLE = True
        except ImportError:
            TOOL_CONFIG_AVAILABLE = False
    except ImportError:
        OLD_SDK_AVAILABLE = False
        TOOL_CONFIG_AVAILABLE = False

# Try importing OpenAI for GPT-4 Responses API
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
print(OPENAI_AVAILABLE)
# Page configuration
st.set_page_config(
    page_title="External Web Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
try:
    from xai_sdk import Client
    from xai_sdk.chat import user, system
    from xai_sdk.search import SearchParameters  
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
@dataclass
class SearchResult:
    success: bool
    response: str
    sources: List[Dict[str, str]]
    search_queries: List[str]
    model: str
    timestamp: str
    response_time: float
    error: Optional[str] = None
    has_grounding: bool = False
    raw_metadata: Optional[Dict] = None

class GeminiGroundingSearch:
    """Optimized Gemini search with reduced latency using only Gemini 2.5 Flash"""
    
    @staticmethod
    def get_sdk_info():
        """Get information about available SDK"""
        if NEW_SDK_AVAILABLE:
            try:
                version = getattr(genai, '__version__', 'Unknown')
                return f"google-genai v{version}", "New SDK (Recommended)", True
            except:
                return "google-genai (Unknown version)", "New SDK (Recommended)", True
        elif OLD_SDK_AVAILABLE:
            try:
                version = getattr(genai_old, '__version__', 'Unknown')
                return f"google-generativeai v{version}", "Legacy SDK", TOOL_CONFIG_AVAILABLE
            except:
                return "google-generativeai (Unknown version)", "Legacy SDK", TOOL_CONFIG_AVAILABLE
        else:
            return "No SDK Available", "None", False
    
    @staticmethod
    def search_with_new_sdk(query: str) -> SearchResult:
        """Optimized search using new SDK with Gemini 2.5 Flash only"""
        start_time = time.time()
        try:
            # Configure client once
            client = genai.Client(api_key=GEMINI_API_KEY)
            
            # Simplified grounding tool
            grounding_tool = types.Tool(google_search=types.GoogleSearch(max_results=1,))
            
            # Minimal config for speed
            config = types.GenerateContentConfig(
                tools=[grounding_tool],
                response_modalities=['TEXT']
            )
            
            # Concise prompt to reduce processing time
            optimized_query = f"""Answer this query with current, accurate information: {query}
            
            Provide key facts and recent developments. Keep response focused and well-structured."""
            
            # Use only Gemini 2.5 Flash (fastest model)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=optimized_query,
                config=config
            )
            
            response_time = time.time() - start_time
            model_used = "gemini-2.5-flash (New SDK)"
            
            # Fast metadata extraction with error handling
            sources = []
            search_queries = []
            has_grounding = False
            
            try:
                if (response.candidates and 
                    hasattr(response.candidates[0], 'grounding_metadata')):
                    
                    metadata = response.candidates[0].grounding_metadata
                    has_grounding = True
                    
                    # Extract search queries
                    if hasattr(metadata, 'web_search_queries'):
                        search_queries = list(metadata.web_search_queries)
                    
                    # Extract sources efficiently
                    if hasattr(metadata, 'grounding_chunks'):
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, 'web') and chunk.web and chunk.web.uri:
                                sources.append({
                                    'title': getattr(chunk.web, 'title', 'Unknown'),
                                    'uri': chunk.web.uri
                                })
            except Exception:
                # Silently continue if metadata extraction fails
                pass
            
            # Extract response text efficiently
            response_text = ""
            if hasattr(response, 'text'):
                response_text = response.text
            elif response.candidates and response.candidates[0].content.parts:
                response_text = ''.join([
                    part.text for part in response.candidates[0].content.parts 
                    if hasattr(part, 'text')
                ])
            
            return SearchResult(
                success=True,
                response=response_text,
                sources=sources,
                search_queries=search_queries,
                model=model_used,
                timestamp=datetime.now().isoformat(),
                response_time=response_time,
                has_grounding=has_grounding
            )
            
        except Exception as e:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="gemini-2.5-flash (Error)",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error=str(e),
                has_grounding=False
            )
    
    @staticmethod
    def search_with_legacy_sdk(query: str) -> SearchResult:
        """Optimized legacy SDK search with Gemini 2.5 Flash only"""
        start_time = time.time()
        try:
            genai_old.configure(api_key=GEMINI_API_KEY)
            
            # Concise prompt for speed
            prompt = f"Provide current, accurate information about: {query}"
            
            # Use only Gemini 2.5 Flash
            model = genai_old.GenerativeModel("gemini-2.5-flash")
            
            # Try grounding first, fallback to basic
            try:
                if TOOL_CONFIG_AVAILABLE:
                    response = model.generate_content(
                        prompt,
                        tools=[{'google_search_retrieval': {}}]
                    )
                    model_used = "gemini-2.5-flash (Legacy Grounding)"
                else:
                    response = model.generate_content(prompt)
                    model_used = "gemini-2.5-flash (Legacy Basic)"
            except Exception:
                # Fallback to basic
                response = model.generate_content(prompt)
                model_used = "gemini-2.5-flash (Legacy Basic)"
            
            response_time = time.time() - start_time
            
            # Fast source extraction
            sources = []
            search_queries = []
            has_grounding = False
            
            try:
                if (response.candidates and 
                    hasattr(response.candidates[0], 'grounding_metadata')):
                    has_grounding = True
                    # Add legacy source extraction if needed
            except Exception:
                pass
            
            return SearchResult(
                success=True,
                response=response.text,
                sources=sources,
                search_queries=search_queries,
                model=model_used,
                timestamp=datetime.now().isoformat(),
                response_time=response_time,
                has_grounding=has_grounding
            )
            
        except Exception as e:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="gemini-2.5-flash (Legacy Error)",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error=str(e),
                has_grounding=False
            )
    
    @staticmethod
    def search(query: str) -> SearchResult:
        """Main search method using the best available SDK"""
        if NEW_SDK_AVAILABLE:
            return GeminiGroundingSearch.search_with_new_sdk(query)
        elif OLD_SDK_AVAILABLE:
            return GeminiGroundingSearch.search_with_legacy_sdk(query)
        else:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="No SDK Available",
                timestamp=datetime.now().isoformat(),
                response_time=0.0,
                error="No Google AI SDK installed. Please install: pip install google-genai",
                has_grounding=False
            )

class GPTResponsesSearch:
    """Handles GPT-4o with OpenAI Responses API for web search"""
    # ...existing code...

    
    @staticmethod
    def extract_response_text(response):
        """
        Extract the actual response text from OpenAI Responses API response
        """
        response_text = ""
        
        try:
            # Method 1: Direct output access (most common)
            if hasattr(response, 'output') and response.output:
                if isinstance(response.output, str):
                    return response.output
                elif hasattr(response.output, 'content'):
                    # Handle content array
                    if isinstance(response.output.content, list):
                        for content_item in response.output.content:
                            if hasattr(content_item, 'text') and content_item.text:
                                response_text += content_item.text
                    elif isinstance(response.output.content, str):
                        response_text = response.output.content
                    else:
                        response_text = str(response.output.content)
                else:
                    response_text = str(response.output)
            
            # Method 2: Try accessing as message format (alternative structure)
            elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                response_text = response.message.content
            
            # Method 3: Try choices format (if it follows chat completion structure)
            elif hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    response_text = choice.message.content
                elif hasattr(choice, 'text'):
                    response_text = choice.text
            
            # Method 4: Last resort - convert to string
            if not response_text:
                response_text = str(response)
        
        except Exception as e:
            print(f"Error extracting response text: {e}")
            response_text = str(response)
        
        return response_text.strip() if response_text else ""

    @staticmethod
    def extract_sources_from_response(response):
        """
        Extract sources/citations from OpenAI Responses API response
        """
        sources = []
        
        try:
            # Method 1: Check for citations attribute
            if hasattr(response, 'citations') and response.citations:
                for i, citation in enumerate(response.citations[:10]):  # Limit to 10
                    source = {
                        'title': citation.get('title', f'Web Source {i+1}'),
                        'uri': citation.get('url', citation.get('uri', ''))
                    }
                    if source['uri']:  # Only add if we have a URL
                        sources.append(source)
            
            # Method 2: Check in metadata
            elif hasattr(response, 'metadata') and response.metadata:
                if hasattr(response.metadata, 'sources'):
                    for i, source in enumerate(response.metadata.sources[:10]):
                        sources.append({
                            'title': source.get('title', f'Web Source {i+1}'),
                            'uri': source.get('url', source.get('uri', ''))
                        })
            
            # Method 3: Check in output metadata
            elif (hasattr(response, 'output') and 
                  hasattr(response.output, 'metadata') and 
                  hasattr(response.output.metadata, 'sources')):
                for i, source in enumerate(response.output.metadata.sources[:10]):
                    sources.append({
                        'title': source.get('title', f'Web Source {i+1}'),
                        'uri': source.get('url', source.get('uri', ''))
                    })
        
        except Exception as e:
            print(f"Error extracting sources: {e}")
        
        return sources
    @staticmethod
    def search(query: str) -> SearchResult:
        """Search using GPT-4o with OpenAI Responses API for web grounding"""
        start_time = time.time()

        if not OPENAI_AVAILABLE:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="OpenAI SDK Not Available",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error="OpenAI SDK not installed. Please install: pip install openai",
                has_grounding=False
            )

        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)

            enhanced_query = f"""
            Please provide comprehensive, current, and accurate information about: "{query}"

            I need detailed information including:
            - Current facts and latest developments
            - Key insights and important details
            - Recent changes or updates (prioritize 2024/2025 information)
            - Multiple perspectives when relevant
            - Specific examples and evidence
            - User location is India

            Please structure your response clearly with proper organization and cite your sources.
            """

            # Use the Responses API
            response = client.responses.create(
                model="gpt-4o",
                instructions="You are a helpful assistant with access to current web information. Always provide accurate, up-to-date information with proper citations when available.",
                input=enhanced_query,
                tools=[{"type": "web_search"}],
                temperature=0.1
            )

            response_time = time.time() - start_time

            # ==============================================================================
            # CORRECT PARSING LOGIC BASED ON YOUR DEBUG OUTPUT
            # ==============================================================================
            
            response_text = ""
            sources = []
            search_queries = [query]
            
            try:
                # Based on your debug output: response.output is a list
                # Item 0: ResponseFunctionWebSearch (web search call)  
                # Item 1: ResponseOutputMessage (actual response content)
                
                if hasattr(response, 'output') and isinstance(response.output, list):
                    
                    # Find the ResponseOutputMessage in the output list
                    for item in response.output:
                        # Check if this is the message response (not the web search call)
                        if (hasattr(item, 'type') and item.type == 'message' and 
                            hasattr(item, 'content') and isinstance(item.content, list)):
                            
                            # Extract text from message content
                            for content_item in item.content:
                                if hasattr(content_item, 'text') and content_item.text:
                                    response_text += content_item.text
                                    
                                    # Extract sources from annotations if available
                                    if hasattr(content_item, 'annotations'):
                                        for annotation in content_item.annotations:
                                            if (hasattr(annotation, 'type') and 
                                                annotation.type == 'url_citation'):
                                                source = {
                                                    'title': getattr(annotation, 'title', 'Web Source'),
                                                    'uri': getattr(annotation, 'url', '')
                                                }
                                                if source['uri'] and source not in sources:
                                                    sources.append(source)
                            break
                    
                    # If no message found, try alternative parsing
                    if not response_text and len(response.output) > 1:
                        # Try the second item (index 1) as it's usually the response
                        second_item = response.output[1]
                        if hasattr(second_item, 'content') and isinstance(second_item.content, list):
                            for content_item in second_item.content:
                                if hasattr(content_item, 'text'):
                                    response_text += content_item.text
                
                # Clean up response text
                response_text = response_text.strip() if response_text else ""
                
                # Remove duplicate sources (keep unique URLs)
                unique_sources = []
                seen_urls = set()
                for source in sources:
                    if source['uri'] not in seen_urls:
                        unique_sources.append(source)
                        seen_urls.add(source['uri'])
                sources = unique_sources[:10]  # Limit to 10 sources
                
            except Exception as parsing_error:
                print(f"Parsing error: {parsing_error}")
                # Fallback to string conversion
                response_text = str(response)
            
            has_grounding = len(sources) > 0 or "web_search" in str(response.output)

            return SearchResult(
                success=True,
                response=response_text,
                sources=sources,
                search_queries=search_queries,
                model="GPT-4o Responses API with Web Search",
                timestamp=datetime.now().isoformat(),
                response_time=response_time,
                has_grounding=has_grounding
            )

        except Exception as e:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="GPT-4 Responses API (Error)",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error=str(e),
                has_grounding=False
            )
class GPTSerperSearch:
    """Handles GPT-4o with Serper API for web search"""
    
    @staticmethod
    def search_web_with_serper(query: str, num_results: int = 10) -> Dict:
        """Search the web using Serper API"""
        try:
            url = "https://google.serper.dev/search"
            payload = {
                'q': query,
                'num': num_results,
                'gl': 'in',  # Country code for India
                'hl': 'en'   # Language
            }
            headers = {
                'X-API-KEY': SERPER_API_KEY,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            print(f"Serper API error: {e}")
            return {}
    
    @staticmethod
    def format_search_results_for_prompt(search_data: Dict) -> str:
        """Format Serper search results for GPT prompt"""
        if not search_data:
            return "No search results available."
        
        formatted_results = []
        
        # Add organic results
        if 'organic' in search_data:
            for i, result in enumerate(search_data['organic'][:8], 1):  # Limit to 8 results
                title = result.get('title', 'No title')
                snippet = result.get('snippet', 'No description')
                link = result.get('link', '')
                
                formatted_results.append(f"""
                                        **Source {i}:**
                                        Title: {title}
                                        URL: {link}
                                        Content: {snippet}
                                        """)
        
        # Add knowledge graph if available
        if 'knowledgeGraph' in search_data:
            kg = search_data['knowledgeGraph']
            if 'description' in kg:
                formatted_results.append(f"""
                                        **Knowledge Graph:**
                                        {kg.get('title', '')}: {kg.get('description', '')}
                                        """)
        
        # Add answer box if available
        if 'answerBox' in search_data:
            answer = search_data['answerBox']
            if 'answer' in answer:
                formatted_results.append(f"""
                                        **Featured Answer:**
                                        {answer.get('answer', '')}
                                        """)
        
        return "\n".join(formatted_results) if formatted_results else "No relevant search results found."
    
    @staticmethod
    def extract_sources_from_serper(search_data: Dict) -> List[Dict[str, str]]:
        """Extract sources from Serper search results"""
        sources = []
        
        if 'organic' in search_data:
            for result in search_data['organic'][:10]:  # Limit to 10 sources
                title = result.get('title', 'Web Source')
                uri = result.get('link', '')
                if uri:
                    sources.append({
                        'title': title,
                        'uri': uri
                    })
        
        return sources
    
    @staticmethod
    def search(query: str) -> SearchResult:
        """Search using GPT-4o with Serper API for web search"""
        start_time = time.time()
        
        if not OPENAI_AVAILABLE:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="OpenAI SDK Not Available",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error="OpenAI SDK not installed. Please install: pip install openai",
                has_grounding=False
            )
        
        try:
            # Step 1: Search the web using Serper API
            search_data = GPTSerperSearch.search_web_with_serper(query)
            
            if not search_data:
                raise Exception("Serper API search failed or returned no results")
            
            # Step 2: Format search results for GPT prompt
            search_context = GPTSerperSearch.format_search_results_for_prompt(search_data)
            
            # Step 3: Create enhanced prompt with search context
            enhanced_query = f"""
                                Based on the following web search results for the query "{query}", please provide a comprehensive, accurate, and well-structured response:

                                {search_context}

                                Please provide detailed information including:
                                - Current facts and latest developments from the search results
                                - Key insights and important details
                                - Recent changes or updates (prioritize 2024/2025 information)
                                - Multiple perspectives when relevant from the sources
                                - Specific examples and evidence from the search results
                                - User location context: India

                                Structure your response clearly and reference the sources naturally in your answer. Synthesize the information from multiple sources when possible.
                                    """
            
            # Step 4: Get GPT-4o response
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful research assistant. Provide comprehensive, accurate responses based on the provided web search results. Always synthesize information from multiple sources and maintain factual accuracy."
                    },
                    {
                        "role": "user",
                        "content": enhanced_query
                    }
                ],
                temperature=0.1,
                #max_tokens=2048
            )
            
            response_time = time.time() - start_time
            
            # Step 5: Extract response text
            response_text = ""
            if response.choices and response.choices[0].message:
                response_text = response.choices[0].message.content
            
            # Step 6: Extract sources from search results
            sources = GPTSerperSearch.extract_sources_from_serper(search_data)
            
            # Step 7: Prepare search queries (what was actually searched)
            search_queries = [query]
            if 'related' in search_data:
                search_queries.extend([rel.get('query', '') for rel in search_data['related'][:3]])
            
            return SearchResult(
                success=True,
                response=response_text,
                sources=sources,
                search_queries=search_queries,
                model="GPT-4o with Serper API",
                timestamp=datetime.now().isoformat(),
                response_time=response_time,
                has_grounding=True,
                raw_metadata=search_data
            )
        
        except Exception as e:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="GPT-4o Serper API (Error)",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error=str(e),
                has_grounding=False
            )
class GrokLiveSearch:
    """Handles Grok-4 with Live Search using official xAI SDK"""
    
    @staticmethod
    def search(query: str) -> SearchResult:
        """Search using Grok-4 with Live Search capability"""
        start_time = time.time()
        #response_time=0.0
        
        if not XAI_AVAILABLE:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="xAI SDK Not Available",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error="xAI SDK not installed. Please install: pip install xai-sdk",
                has_grounding=False
            )
        
        try:
            # Initialize xAI client
            client = Client(api_key=XAI_API_KEY)
            
            # Enhanced query for Grok's live search
            enhanced_query = f"""
            Please provide comprehensive, current, and accurate information about: "{query}"
            
            I need detailed information including:
            - Current facts and latest developments 
            - Key insights and important details
            - Recent changes or updates (prioritize 2024/2025 information)
            - Multiple perspectives when relevant
            - Specific examples and evidence
            - User location context: India
            
            Use your live search capabilities to find the most up-to-date information available.
            Please structure your response clearly and cite sources when available.
            """
            
            # Configure search parameters for live search
            search_params = SearchParameters(
                mode="auto",  # Let Grok decide when to search
                return_citations=True#,  # Include citations
                #max_results=20,  # Maximum search results to consider
                #sources=["web", "x"]  # Search both web and X/Twitter
            )
            
            # Make request with live search enabled
            from xai_sdk.chat import user
            response = client.chat.create(
                    model="grok-4-0709",
                    messages=[
                        user(enhanced_query)
                    ],
                    search_parameters=SearchParameters(
                    mode="on",
                    return_citations=True,
                    ),
                    temperature=0.1
                    )
            
            #response = requests.post(
                #"https://api.x.ai/v1/chat/completions",
                #headers={
                    #"Authorization": f"Bearer {XAI_API_KEY}",
                    #"Content-Type": "application/json"
                #},
                #json={
                    #"model": "grok-4-0709",  # Use the correct model name
                    #"messages": [
                        #{"role": "system", "content": "You are Grok, a helpful AI assistant with access to real-time information through web search. Use your search capabilities to provide current, accurate information."},
                        #{"role": "user", "content": enhanced_query}
                    #],
                    #"stream": False,
                    #"temperature": 0.1,
                    #"max_tokens": 4000,
                    #"tools": [{
                        #"type": "function",
                        #"function": {
                            #"name": "live_search",
                            #"description": "Search the web for current information",
                            #"parameters": {
                                #"type": "object",
                                #"properties": {
                                    #"query": {
                                        #"type": "string",
                                        #"description": "The search query"
                                    #},
                                    #"sources": {
                                        #"type": "array",
                                        #"items": {"type": "string"},
                                        #"description": "Sources to search"
                                    #}
                            # },
                                #"required": ["query"]
                            #}
                        #}
                    #}],  # This enables live search
                    #"tool_choice": "auto"  # Let Grok decide when to search
                #},
                #timeout=180
            #)
            # After: response = client.chat.create(...)

# Extract response text and sources directly from the SDK response object
            # Extract Grok response text (actual answer)
            response_text = ""
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    response_text = choice.message.content
                elif hasattr(choice, "text"):
                    response_text = choice.text
            else:
                response_text = str(response)
            #print(response)

            sources = []
            search_queries = [query]
            has_grounding = False

            # If the SDK provides citations or sources, extract them here
            if hasattr(response, "citations"):
                for citation in response.citations:
                    sources.append({
                        "title": citation.get("title", "Web Source"),
                        "uri": citation.get("url", citation.get("uri", ""))
                    })
                has_grounding = bool(sources)

                        # Continue with your logic...
                        
            response_time = time.time() - start_time
                            
                            # Extract response text
                            # Extract response text from HTTP response
                            # Extract response text and sources directly from the SDK response object
            response_text = getattr(response, "text", str(response))
            sources = []
            search_queries = [query]
            has_grounding = False

                        # If the SDK provides citations or sources, extract them here
            if hasattr(response, "citations"):
                for citation in response.citations:
                    sources.append({
                                    "title": citation.get("title", "Web Source"),
                                    "uri": citation.get("url", citation.get("uri", ""))
                                })
                has_grounding = bool(sources)

# If no sources found, try to extract from response text
            if not sources:
                sources = GrokLiveSearch.extract_sources_from_response(response_text)

            # If no citations found but response seems to have web data, mark as grounded
            if not has_grounding and (len(sources) > 0 or "according to" in response_text.lower() or "source:" in response_text.lower()):
                has_grounding = True
                        # Extract response text from HTTP response
                        
                        # Extract sources from response and usage data
            
            
            
            if not sources:
                sources = GrokLiveSearch.extract_sources_from_response(response_text)
            
            # If no citations found but response seems to have web data, mark as grounded
            if not has_grounding and (len(sources) > 0 or "according to" in response_text.lower() or "source:" in response_text.lower()):
                has_grounding = True
            
            return SearchResult(
                success=True,
                response=response_text,
                sources=sources, 
                search_queries=search_queries,
                model="Grok-4 with Live Search (xAI SDK)",
                timestamp=datetime.now().isoformat(),
                response_time=response_time,
                has_grounding=has_grounding,
                raw_metadata=response
            )
            
        except Exception as e:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="Grok-4 Live Search (Error)",
                timestamp=datetime.now().isoformat(), 
                response_time=time.time() - start_time,
                error=str(e),
                has_grounding=False,
                raw_metadata=response
            )
    
    @staticmethod
    def extract_sources_from_response(response_text: str) -> List[Dict[str, str]]:
        """Extract sources from Grok response text as fallback"""
        sources = []
        
        try:
            import re
            
            # Common patterns for extracting sources from Grok responses
            patterns = [
                # URLs in markdown format [title](url)
                r'\[([^\]]+)\]\((https?://[^\)]+)\)',
                # Direct URLs
                r'(https?://[^\s\]\)]+)',
                # Source citations
                r'Source:\s*([^\n]+)',
                r'According to\s+([^,\n]+)',
                r'From\s+([^,\n]+)',
            ]
            
            found_urls = set()
            
            # Extract markdown links first (best quality)
            markdown_links = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', response_text)
            for title, url in markdown_links:
                if url not in found_urls and len(sources) < 10:
                    sources.append({
                        'title': title.strip(),
                        'uri': url.strip()
                    })
                    found_urls.add(url)
            
            # Extract direct URLs
            direct_urls = re.findall(r'https?://[^\s\]\)]+', response_text)
            for url in direct_urls:
                if url not in found_urls and len(sources) < 10:
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        title = domain.replace('www.', '').replace('.com', '').title()
                    except:
                        title = f"Web Source {len(sources) + 1}"
                    
                    sources.append({
                        'title': title,
                        'uri': url.strip()
                    })
                    found_urls.add(url)
        
        except Exception as e:
            print(f"Error extracting sources from Grok response: {e}")
        
        return sources[:10]
    
class GPTChatCompletionsWebSearch:
    """Handles GPT-4o with Chat Completions API using function calling for web search"""
    
    @staticmethod
    def search_web_with_serper(query: str, num_results: int = 10) -> Dict:
        """Search the web using Serper API - reuse from GPTSerperSearch"""
        try:
            url = "https://google.serper.dev/search"
            payload = {
                'q': query,
                'num': num_results,
                'gl': 'in',  # Country code for India
                'hl': 'en'   # Language
            }
            headers = {
                'X-API-KEY': SERPER_API_KEY,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            print(f"Serper API error: {e}")
            return {}
    
    @staticmethod
    def web_search_function(query: str) -> str:
        """Function to be called by GPT-4o for web search"""
        search_data = GPTChatCompletionsWebSearch.search_web_with_serper(query)
        
        if not search_data:
            return "No search results found."
        
        # Format results more concisely for function response
        results = []
        
        # Add organic results
        if 'organic' in search_data:
            for result in search_data['organic'][:5]:  # Limit to top 5 for function response
                title = result.get('title', 'No title')
                snippet = result.get('snippet', 'No description')
                link = result.get('link', '')
                results.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}\n")
        
        # Add knowledge graph if available
        if 'knowledgeGraph' in search_data:
            kg = search_data['knowledgeGraph']
            if 'description' in kg:
                results.append(f"Knowledge Graph: {kg.get('title', '')}: {kg.get('description', '')}\n")
        
        # Add answer box if available
        if 'answerBox' in search_data:
            answer = search_data['answerBox']
            if 'answer' in answer:
                results.append(f"Featured Answer: {answer.get('answer', '')}\n")
        
        return "\n".join(results[:10])  # Return top results
    
    @staticmethod
    def search(query: str) -> SearchResult:
        """Search using GPT-4o with Chat Completions API and function calling"""
        start_time = time.time()
        
        if not OPENAI_AVAILABLE:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="OpenAI SDK Not Available",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error="OpenAI SDK not installed. Please install: pip install openai",
                has_grounding=False
            )
        
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            # Define the web search function
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web for current information using Google search",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to find current information"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
            
            # Enhanced system message
            system_message = """You are a helpful research assistant with access to web search capabilities. When a user asks for information that might benefit from current web data, use the web_search function to find up-to-date information. Always provide comprehensive, accurate responses based on the search results and cite your sources when possible."""
            
            # Enhanced user query
            enhanced_query = f"""
            Please provide comprehensive, current, and accurate information about: "{query}"
            
            I need detailed information including:
            - Current facts and latest developments
            - Key insights and important details  
            - Recent changes or updates (prioritize 2024/2025 information)
            - Multiple perspectives when relevant
            - Specific examples and evidence
            - User location context: India
            
            Please use web search to find the most current information available and structure your response clearly.
            """
            
            # First API call - let GPT decide if it needs to search
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": enhanced_query}
            ]
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
                tool_choice="auto",  # Let GPT decide when to use tools
                temperature=0.1,
                max_tokens=4000
            )
            
            # Store sources and search queries
            sources = []
            search_queries = [query]
            
            # Check if GPT wants to use function calling
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            
            if tool_calls:
                # GPT decided to search - execute function calls
                messages.append(response_message)  # Add GPT's message with tool calls
                
                for tool_call in tool_calls:
                    if tool_call.function.name == "web_search":
                        # Parse the search query from function call
                        function_args = json.loads(tool_call.function.arguments)
                        search_query = function_args.get("query", query)
                        search_queries.append(search_query)
                        
                        # Execute the web search
                        search_results = GPTChatCompletionsWebSearch.web_search_function(search_query)
                        
                        # Add function response to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": search_results
                        })
                        
                        # Extract sources from the original search data
                        search_data = GPTChatCompletionsWebSearch.search_web_with_serper(search_query)
                        if 'organic' in search_data:
                            for result in search_data['organic'][:10]:
                                title = result.get('title', 'Web Source')
                                uri = result.get('link', '')
                                if uri:
                                    sources.append({'title': title, 'uri': uri})
                
                # Second API call with search results
                final_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4000
                )
                
                response_text = final_response.choices[0].message.content
                has_grounding = True
                
            else:
                # GPT didn't use function calling - use original response
                response_text = response_message.content
                has_grounding = False
            
            response_time = time.time() - start_time
            
            return SearchResult(
                success=True,
                response=response_text,
                sources=sources,
                search_queries=list(set(search_queries)),  # Remove duplicates
                model="GPT-4o Chat Completions with Function Calling",
                timestamp=datetime.now().isoformat(),
                response_time=response_time,
                has_grounding=has_grounding
            )
        
        except Exception as e:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="GPT-4o Chat Completions (Error)",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error=str(e),
                has_grounding=False
            )

def add_citations_to_text(response_result: SearchResult) -> str:
    """Add inline citations to the response text"""
    if not response_result.has_grounding or not response_result.sources:
        return response_result.response
    
    text = response_result.response
    
    # Simple citation approach - add numbered references at the end
    if response_result.sources:
        citations = []
        for i, source in enumerate(response_result.sources, 1):
            title = source.get('title', f'Source {i}')
            uri = source.get('uri', '')
            if uri:
                citations.append(f"[{i}] {title}: {uri}")
        
        if citations:
            text += "\n\n**Sources:**\n" + "\n".join(citations)
    
    return text

def display_search_result(result: SearchResult):
    """Display search results with proper grounding information"""
    if result.success:
        # Success header with metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⚡ Response Time", f"{result.response_time:.2f}s")
        with col2:
            st.metric("🔗 Sources", len(result.sources))
        with col3:
            st.metric("🔍 Searches", len(result.search_queries))
        with col4:
            st.metric("🌐 Grounded", "Yes" if result.has_grounding else "No")
        
        # Show grounding status
        if result.has_grounding:
            st.success(f"✅ Successfully grounded search: {result.model}")
        else:
            st.warning(f"⚠️ No grounding available: {result.model}")
        
        # Show search queries used
        if result.search_queries:
            st.info(f"🔍 **Search queries used:** {', '.join(result.search_queries)}")
        
        # Main response
        st.subheader("📄 Search Response")
        
        # Add citations to response
        response_with_citations = add_citations_to_text(result)
        st.markdown(response_with_citations)
        
        # Sources section (if available)
        if result.sources:
            st.subheader("🔗 Sources & References")
            for i, source in enumerate(result.sources, 1):
                title = source.get('title', f'Source {i}')
                uri = source.get('uri', '')
                if uri:
                    st.markdown(f"**{i}.** [{title}]({uri})")
                else:
                    st.markdown(f"**{i}.** {title}")
        
        # Technical details
        with st.expander("📊 Technical Details"):
            st.json({
                "timestamp": result.timestamp,
                "model_used": result.model,
                "response_time_seconds": result.response_time,
                "sources_count": len(result.sources),
                "search_queries_count": len(result.search_queries),
                "has_grounding": result.has_grounding
            })
    
    else:
        st.error(f"❌ Search failed")
        
        with st.expander("🔧 Error Details & Solutions", expanded=True):
            st.error(f"**Error:** {result.error}")
            
            error_str = str(result.error or "").upper()
            
            if "NO GOOGLE AI SDK" in error_str or "NO SDK" in error_str:
                st.markdown("""
                **📦 SDK Installation Required:**
                
                For Gemini grounding experience, install the new SDK:
                ```bash
                pip install google-genai
                ```
                
                Or install the legacy SDK:
                ```bash
                pip install google-generativeai
                ```
                """)
            
            elif "OPENAI" in error_str:
                st.markdown("""
                **📦 OpenAI SDK Installation Required:**
                
                For GPT-4 Responses API:
                ```bash
                pip install openai
                ```
                """)
            
            elif "API_KEY" in error_str or "INVALID" in error_str or "AUTHENTICATION" in error_str:
                st.markdown("""
                **🔑 API Key Issues:**
                1. API keys are embedded in the code
                2. Contact administrator if authentication fails
                3. Check if API access is enabled for your service
                """)
            
            elif "PERMISSION" in error_str or "FORBIDDEN" in error_str:
                st.markdown("""
                **🚫 Permission Issues:**
                1. API key may not have required permissions
                2. Grounding may not be available in your region
                3. Check service settings and quotas
                """)
            
            else:
                st.markdown("""
                **🔧 General Troubleshooting:**
                1. Update to latest SDKs
                2. Check internet connection
                3. Verify service status
                4. Try a simpler query first
                """)

    #st.subheader("🛠️ Raw Model Response (Debug)")
    #st.code(str(result.raw_metadata if hasattr(result, "raw_metadata") else result.response), language="python")

def main():
    # Header
    st.title("🔍Websearch Comparison")
    st.markdown("**Choose between Gemini 2.5/2.0 Flash Grounding, or GPT-4o Responses API, or GPT-4o with Serper API, or Grok4 with Live search**")
    
    # Model Selection
    st.subheader("🤖 Select AI Model")
    
    # Check SDK availability
    gemini_available = NEW_SDK_AVAILABLE or OLD_SDK_AVAILABLE
    openai_available = OPENAI_AVAILABLE
    serper_available = bool(SERPER_API_KEY and SERPER_API_KEY != "your-serper-api-key-here")
    grok_available = XAI_AVAILABLE and bool(XAI_API_KEY and XAI_API_KEY != "your-xai-api-key-here")
    
    options = []
    if gemini_available:
        options.append("Gemini 2.5/2.0 Flash with Google Search Grounding")
    if openai_available:
        options.append("GPT-4o with Responses API Web Search")
    if openai_available and serper_available:
        options.append("GPT-4o with Serper API Web Search")
    if grok_available:
        options.append("Grok-4 with Live Web Search")
    if openai_available:

        options.append("GPT-4o with Chat Completions API")  # Add this line

    
    
    if not options:
        st.error("❌ No AI models available. Please install required SDKs and configure API keys:")
        st.markdown("""
        ```bash
        pip install google-genai openai requests
        ```
        """)
        return
    
    selected_model = st.selectbox(
        "Choose your AI model:",
        options,
        index=0
    )
    
    # Show model status
    if "Gemini" in selected_model:
        sdk_info, sdk_type, has_support = GeminiGroundingSearch.get_sdk_info()
        if NEW_SDK_AVAILABLE:
            st.success(f"✅ {sdk_info} - Full grounding support enabled")
        elif OLD_SDK_AVAILABLE and has_support:
            st.warning(f"⚠️ {sdk_info} - Limited grounding support")
        else:
            st.error(f"❌ {sdk_info}")
    elif "Responses API" in selected_model:
        if openai_available:
            st.success("✅ OpenAI SDK available - GPT-4o with web search enabled")
        else:
            st.error("❌ OpenAI SDK not available")
    elif "Serper API" in selected_model:
        if openai_available and serper_available:
            st.success("✅ GPT-4o with Serper API - Full web search enabled")
        elif not openai_available:
            st.error("❌ OpenAI SDK not available")
        elif not serper_available:
            st.error("❌ Serper API key not configured")
    elif "Grok" in selected_model:
        if grok_available:
            st.success("✅ xAI Grok SDK available - Live search enabled")
        else:
            st.error("❌ xAI SDK not installed or API key not configured")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # API Key status
    st.sidebar.subheader("🔑 API Keys")
    st.sidebar.success("✅ Gemini API Key: Configured")
    if OPENAI_API_KEY and OPENAI_API_KEY != "sk-your-openai-api-key-here":
        st.sidebar.success("✅ OpenAI API Key: Configured")
    else:
        st.sidebar.warning("⚠️ OpenAI API Key: Not Configured")
    
    if SERPER_API_KEY and SERPER_API_KEY != "your-serper-api-key-here":
        st.sidebar.success("✅ Serper API Key: Configured")
    else:
        st.sidebar.warning("⚠️ Serper API Key: Not Configured")
    if XAI_API_KEY and XAI_API_KEY != "your-xai-api-key-here":
        st.sidebar.success("✅ xAI Grok API Key: Configured")
    else:
        st.sidebar.warning("⚠️ xAI Grok API Key: Not Configured")
    # Settings
    st.sidebar.subheader("⚙️ Settings")
    save_history = st.sidebar.checkbox("💾 Save History", True)
    show_citations = st.sidebar.checkbox("📎 Show Citations", True)
    
    # Main interface
    st.subheader("🔍 Search Query")
    
    # Search input
    col1, col2 = st.columns([5, 1])
    with col1:
        search_query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., Latest AI developments in 2025",
            key="main_search",
            label_visibility="collapsed"
        )
    with col2:
        st.write("")
        search_button = st.button("🚀 Search", type="primary", use_container_width=True)
    
    
    # Execute search
    if search_button and search_query:
        if not search_query.strip():
            st.warning("⚠️ Please enter a search query")
            return
        
        # Determine which model to use
        if "Gemini" in selected_model:
            if not gemini_available:
                st.error("❌ Gemini SDK not available")
                return
            
            with st.spinner(f"🔍 Searching with Gemini..."):
                result = GeminiGroundingSearch.search(search_query)
        
        elif "Responses API" in selected_model:  # GPT-4 Responses API
            if not openai_available:
                st.error("❌ OpenAI SDK not available")
                return
            
            if OPENAI_API_KEY == "sk-your-openai-api-key-here":
                st.error("❌ OpenAI API key not configured in code")
                return
            
            with st.spinner(f"🔍 Searching with GPT-4o Responses API..."):
                result = GPTResponsesSearch.search(search_query)

        elif "Serper API" in selected_model:  # GPT-4o with Serper
            if not openai_available:
                st.error("❌ OpenAI SDK not available")
                return
            
            if not serper_available:
                st.error("❌ Serper API key not configured")
                return
            
            with st.spinner(f"🔍 Searching with GPT-4o + Serper API..."):
                result = GPTSerperSearch.search(search_query)
        elif "Grok" in selected_model:  # Grok-4 with Live Search
            if not grok_available:
                st.error("❌ xAI Grok SDK not available or API key not configured")
                return
            
            with st.spinner(f"🔍 Searching with Grok-4 Live Search..."):
                result = GrokLiveSearch.search(search_query)

        elif "Function Calling" in selected_model:  # GPT-4o Chat Completions
            if not openai_available:
                st.error("❌ OpenAI SDK not available")
                return
            
            if not serper_available:
                st.error("❌ Serper API key not configured")
                return
            
            with st.spinner(f"🔍 Searching with GPT-4o Chat Completions..."):
                result = GPTChatCompletionsWebSearch.search(search_query)
                
        st.divider()
        display_search_result(result)
        
        # Save to history
        if save_history and result.success:
            if 'search_history' not in st.session_state:
                st.session_state.search_history = []
            
            st.session_state.search_history.append({
                'query': search_query,
                'result': result,
                'model': selected_model,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep last 10
            if len(st.session_state.search_history) > 10:
                st.session_state.search_history = st.session_state.search_history[-10:]
    
    elif search_button and not search_query:
        st.warning("🔍 Please enter a search query")
    
    # History sidebar
    if save_history and 'search_history' in st.session_state and st.session_state.search_history:
        st.sidebar.subheader("📝 Recent Searches")
        
        for i, item in enumerate(reversed(st.session_state.search_history[-5:])):
            with st.sidebar.expander(f"🔍 {item['query'][:25]}..."):
                st.write(f"**Model:** {item['model'][:20]}...")
                st.write(f"**Time:** {item['timestamp'][:16]}")
                st.write(f"**Grounded:** {'Yes' if item['result'].has_grounding else 'No'}")
                st.write(f"**Sources:** {len(item['result'].sources)}")
                if st.button("🔄 Rerun", key=f"rerun_{i}"):
                    st.session_state.main_search = item['query']
                    st.rerun()
        
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.search_history = []
            st.rerun()
    
    # Footer
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **🔍 Available Models:**
        - Gemini: {'Available' if gemini_available else 'Not Available'}
        - GPT-4o: {'Available' if openai_available else 'Not Available'}
        - GPT-4o Serper: {'Available' if (openai_available and serper_available) else 'Not Available'}
        - Grok-4: {'Available' if grok_available else 'Not Available'}
        - Grounding: All support web search
        - Current: {selected_model[:30]}...
        """)
    
    with col2:
        st.markdown("""
        **💡 Tips:**
        - Use specific queries for better results
        - Include current year for recent info
        - All models provide real-time data
        - Grok has access to X/Twitter data
        - Live Search costs $25 per 1,000 sources
        - Check sources for verification
        """)
    
    st.markdown("---")
    st.markdown(
        "**Powered by Gemini , GPT-4o,Grok-4 and Serper API with Web Search** | "
        "API Keys Embedded | "
        "Usage may incur costs"
    )

if __name__ == "__main__":
    main()