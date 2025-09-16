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
    """Handles Gemini with proper Google Search Grounding using official methods"""
    
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
        """Search using the new google-genai SDK with proper grounding (Official Method)"""
        start_time = time.time()
        try:
            # Configure the client with API key
            client = genai.Client(api_key=GEMINI_API_KEY)
            
            # Define the grounding tool (official method)
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            # Configure generation settings
            config = types.GenerateContentConfig(
                tools=[grounding_tool],
                response_modalities=['TEXT']
            )
            
            # Enhanced prompt for better results
            enhanced_query = f"""
            Please provide comprehensive, current, and accurate information about: "{query}"
            
            I need detailed information including:
            - Current facts and latest developments
            - Key insights and important details
            - Recent changes or updates (prioritize 2024/2025 information)
            - User location is India
            - Multiple perspectives when relevant
            - Specific examples and evidence
            
            Please structure your response clearly with proper organization.
            """
            
            # Try different models in order of preference (only 2.5 and 2.0)
            models_to_try = [
                "gemini-2.5-flash",
                "gemini-2.5-pro", 
                "gemini-2.0-flash-exp"
            ]
            
            response = None
            model_used = None
            last_error = None
            
            for model_name in models_to_try:
                try:
                    # Make the request with grounding
                    response = client.models.generate_content(
                        model=model_name,
                        contents=enhanced_query,
                        config=config
                    )
                    model_used = f"{model_name} (New SDK Grounding)"
                    break
                except Exception as model_error:
                    last_error = str(model_error)
                    continue
            
            if response is None:
                raise Exception(f"All models failed. Last error: {last_error}")
            
            response_time = time.time() - start_time
            
            # Extract grounding metadata (official structure)
            sources = []
            search_queries = []
            has_grounding = False
            raw_metadata = None
            
            try:
                if (hasattr(response, 'candidates') and 
                    response.candidates and 
                    hasattr(response.candidates[0], 'grounding_metadata')):
                    
                    grounding_metadata = response.candidates[0].grounding_metadata
                    has_grounding = True
                    raw_metadata = grounding_metadata
                    
                    # Extract search queries used
                    if hasattr(grounding_metadata, 'web_search_queries'):
                        search_queries = list(grounding_metadata.web_search_queries)
                    
                    # Extract grounding chunks (sources)
                    if hasattr(grounding_metadata, 'grounding_chunks'):
                        for chunk in grounding_metadata.grounding_chunks:
                            if hasattr(chunk, 'web') and chunk.web:
                                source_info = {
                                    'title': getattr(chunk.web, 'title', 'Unknown'),
                                    'uri': getattr(chunk.web, 'uri', ''),
                                }
                                if source_info['uri']:  # Only add if URI exists
                                    sources.append(source_info)
                
            except Exception as metadata_error:
                # Grounding metadata extraction failed, but response succeeded
                pass
            
            # Get response text
            response_text = ""
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    response_text = ''.join([
                        part.text for part in candidate.content.parts 
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
                has_grounding=has_grounding,
                raw_metadata=raw_metadata
            )
            
        except Exception as e:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="Gemini (Error)",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error=str(e),
                has_grounding=False
            )
    
    @staticmethod
    def search_with_legacy_sdk(query: str) -> SearchResult:
        """Fallback search using legacy google-generativeai SDK"""
        start_time = time.time()
        try:
            genai_old.configure(api_key=GEMINI_API_KEY)
            
            enhanced_prompt = f"""
            Please provide comprehensive, current information about: "{query}"
            Include recent developments, key facts, and detailed insights.
            Structure your response clearly with proper organization.
            """
            
            model_used = "Legacy SDK"
            
            # Try grounding with legacy SDK if available - only 2.5/2.0 models
            if TOOL_CONFIG_AVAILABLE:
                try:
                    # Try Gemini 2.5 first
                    try:
                        model = genai_old.GenerativeModel("gemini-2.5-flash")
                        response = model.generate_content(
                            enhanced_prompt,
                            tools=[{'google_search_retrieval': {}}]
                        )
                        model_used = "Gemini 2.5 Flash (Legacy Grounding)"
                    except:
                        # Fallback to 2.0
                        model = genai_old.GenerativeModel("gemini-2.0-flash-exp")
                        response = model.generate_content(
                            enhanced_prompt,
                            tools=[{'google_search_retrieval': {}}]
                        )
                        model_used = "Gemini 2.0 Flash (Legacy Grounding)"
                    
                except Exception as e:
                    # Fallback to basic model
                    model = genai_old.GenerativeModel("gemini-2.5-flash")
                    response = model.generate_content(enhanced_prompt)
                    model_used = "Gemini 2.5 Flash (Legacy Basic)"
            else:
                # Basic model without grounding
                model = genai_old.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(enhanced_prompt)
                model_used = "Gemini 2.5 Flash (No Grounding)"
            
            response_time = time.time() - start_time
            
            # Try to extract sources from legacy response
            sources = []
            search_queries = []
            has_grounding = False
            
            try:
                if (hasattr(response, 'candidates') and response.candidates and 
                    hasattr(response.candidates[0], 'grounding_metadata')):
                    has_grounding = True
                    # Legacy source extraction logic here
                    pass
            except:
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
                model="Legacy SDK (Error)",
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
            #response = client.chat.create(
                #model="grok-4-0709",  # Latest Grok-4 model
                #messages=enhanced_query,
                #temperature=0.1,
                #max_tokens=2048,
                #search_parameters=search_params  # Enable live search
            #)
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {XAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-beta",  # Use the correct model name
                    "messages": [
                        {"role": "system", "content": "You are Grok, a helpful AI assistant with access to real-time information through web search. Use your search capabilities to provide current, accurate information."},
                        {"role": "user", "content": enhanced_query}
                    ],
                    "stream": False,
                    "temperature": 0.1,
                    "max_tokens": 4000,
                    "tools": [{"type": "web_search"}],  # This enables live search
                    "tool_choice": "auto"  # Let Grok decide when to search
                },
                timeout=180
            )
            if response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")

            if not response.text.strip():
                raise Exception("Empty response from API")

            try:
                response_data = response.json()
            except ValueError as e:
                raise Exception(f"Invalid JSON response: {response.text[:500]}")
            
            response_time = time.time() - start_time
            
            # Extract response text
            # Extract response text from HTTP response
            
            # Extract response text from HTTP response
            response_text = ""

            # Check if response_data is a string (direct text response)
            if isinstance(response_data, str):
                response_text = response_data
            # Check if it's a dict with standard chat completion format
            elif isinstance(response_data, dict):
                if 'choices' in response_data and response_data['choices']:
                    response_text = response_data['choices'][0].get('message', {}).get('content', '')
                elif 'content' in response_data:
                    response_text = response_data['content']
                else:
                    response_text = str(response_data)
            else:
                response_text = str(response_data)
            
            # Extract sources from response and usage data
            sources = []
            search_queries = [query]
            has_grounding = False
            
            # Extract usage information if available
            # Extract usage information if available
            #usage = response_data.get('usage', {})
            #has_grounding = usage.get('num_sources_used', 0) > 0
            # Extract usage information if available
            if isinstance(response_data, dict):
                usage = response_data.get('usage', {})
                has_grounding = usage.get('num_sources_used', 0) > 0
            else:
                has_grounding = False
            
            # Extract citations from response
            # Extract citations from response
            #citations = response_data.get('citations', [])
            # Extract citations from response
            if isinstance(response_data, dict):
                citations = response_data.get('citations', [])
            else:
                citations = []
            for citation in citations:
                source = {
                    'title': citation.get('title', 'Web Source'),
                    'uri': citation.get('url', citation.get('uri', ''))
                }
                if source['uri']:
                    sources.append(source)
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
                has_grounding=has_grounding
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
                has_grounding=False
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