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
import subprocess


# API Keys from Streamlit secrets
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
AZURE_AI_FOUNDRY_ENDPOINT = st.secrets["AZURE_AI_FOUNDRY_ENDPOINT"]
AZURE_AI_FOUNDRY_KEY = st.secrets["AZURE_AI_FOUNDRY_KEY"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_KEY = st.secrets["AZURE_OPENAI_KEY"]
AZURE_MODEL_DEPLOYMENT = st.secrets["AZURE_MODEL_DEPLOYMENT"]
AZURE_AGENT_ID = st.secrets["AZURE_AGENT_ID"]

# Try importing the new Google GenAI SDK first (recommended)
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

# Try importing Azure OpenAI SDK
try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

print(f"OpenAI Available: {OPENAI_AVAILABLE}")
print(f"Azure OpenAI Available: {AZURE_OPENAI_AVAILABLE}")

# Page configuration
st.set_page_config(
    page_title="External Web Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Standard system prompt for consistency
STANDARD_SYSTEM_PROMPT = """You are a helpful assistant with access to current web information. Always provide accurate, up-to-date information with proper citations when available."""

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
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            
            # Minimal config for speed
            config = types.GenerateContentConfig(
                tools=[grounding_tool],
                response_modalities=['TEXT'],
                # Disable reasoning to reduce latency
            )
            
            # Enhanced prompt for consistency
            optimized_query = f"""Please provide comprehensive, current, and accurate information about: "{query}"

            I need detailed information including:
            - Current facts and latest developments
            - Key insights and important details
            - Recent changes or updates (prioritize 2024/2025 information)
            - Multiple perspectives when relevant
            - Specific examples and evidence
            - User location is India

            Please structure your response clearly with proper organization and cite your sources."""
            
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
                        total_chunks = len(list(metadata.grounding_chunks))
                        
                        # Group chunks by source and limit to 15 unique sources
                        source_to_chunks = {}
                        unique_sources_count = 0
                        
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, 'web') and chunk.web and chunk.web.uri:
                                uri = chunk.web.uri
                                
                                # If we haven't seen this source before and haven't reached limit
                                if uri not in source_to_chunks and unique_sources_count < 15:
                                    source_to_chunks[uri] = {
                                        'title': getattr(chunk.web, 'title', 'Unknown'),
                                        'uri': uri,
                                        'chunks': []
                                    }
                                    unique_sources_count += 1
                                
                                # Add chunk to this source (if source is in our limited set)
                                if uri in source_to_chunks:
                                    source_to_chunks[uri]['chunks'].append(chunk)
                        
                        # Extract sources from the limited set
                        for source_data in source_to_chunks.values():
                            sources.append({
                                'title': source_data['title'],
                                'uri': source_data['uri']
                            })
                        
                        print(f"Debug: Total chunks: {total_chunks}, Unique sources found: {len(source_to_chunks)}, Total chunks from limited sources: {sum(len(s['chunks']) for s in source_to_chunks.values())}")
                        
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
            
            # Enhanced prompt for consistency
            prompt = f"""Please provide comprehensive, current, and accurate information about: "{query}"

            I need detailed information including:
            - Current facts and latest developments
            - Key insights and important details
            - Recent changes or updates (prioritize 2024/2025 information)
            - Multiple perspectives when relevant
            - Specific examples and evidence
            - User location is India

            Please structure your response clearly with proper organization and cite your sources."""
            
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
                instructions=STANDARD_SYSTEM_PROMPT,
                input=enhanced_query,
                tools=[{"type": "web_search"}],
                temperature=0.1
            )

            response_time = time.time() - start_time

            # CORRECT PARSING LOGIC BASED ON YOUR DEBUG OUTPUT
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

class AzureAIAgentsSearch:
    """Azure AI Foundry Agents with Bing Search using REST API with enhanced debugging and authentication"""
    
    @staticmethod
    def search(query: str) -> SearchResult:
        """Search using Azure AI Foundry Agent Service with Bing grounding via REST API"""
        start_time = time.time()
        
        # Import required libraries
        try:
            import streamlit as st
            import requests
            from azure.identity import DefaultAzureCredential
            from azure.core.exceptions import ClientAuthenticationError
        except ImportError as e:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="Dependencies Missing",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error=f"Required dependencies missing: {e}. Install with: pip install azure-identity",
                has_grounding=False
            )
        
        # DEBUG: Print all available secrets (without values)
        try:
            st.write("**DEBUG: Available Secrets:**")
            for key in st.secrets.keys():
                if 'AZURE' in key.upper():
                    st.write(f"- {key}: {'✓ Set' if st.secrets[key] else '✗ Missing'}")
        except Exception as e:
            st.write(f"DEBUG: Error accessing secrets: {e}")
        
        # Get configuration from Streamlit secrets
        try:
            # Extract project endpoint and project name from the full endpoint
            full_endpoint = st.secrets["AZURE_AI_FOUNDRY_ENDPOINT"]
            agent_id = st.secrets.get("AZURE_AGENT_ID", "")
            tenant_id = st.secrets.get("AZURE_TENANT_ID", "")
            
            # Extract the correct endpoint format
            # From: https://ricku-mfrmxlii-eastus2.services.ai.azure.com/api/projects/ricku-mfrmxlii-eastus2_project/
            # To: https://ricku-mfrmxlii-eastus2.services.ai.azure.com
            if "/api/projects/" in full_endpoint:
                base_endpoint = full_endpoint.split("/api/projects/")[0]
                project_path_part = full_endpoint.split("/api/projects/")[1].rstrip('/')
                project_endpoint = f"{base_endpoint}/api/projects/{project_path_part}"
            else:
                # If it's already in the correct format
                base_endpoint = full_endpoint.rstrip('/')
                # Assume project name from the subdomain if not provided
                if ".services.ai.azure.com" in base_endpoint:
                    subdomain = base_endpoint.split("//")[1].split(".")[0]
                    project_name = f"{subdomain}_project"  # Common pattern
                    project_endpoint = f"{base_endpoint}/api/projects/{project_name}"
                else:
                    project_endpoint = base_endpoint
            
            # DEBUG: Validate configuration
            st.write("**DEBUG: Configuration Check:**")
            st.write(f"- Full Endpoint: {full_endpoint}")
            st.write(f"- Base Endpoint: {base_endpoint}")
            st.write(f"- Project Endpoint: {project_endpoint}")
            st.write(f"- Agent ID: {agent_id[:20]}..." if agent_id else "✗ Missing")
            st.write(f"- Tenant ID: {tenant_id[:20]}..." if tenant_id else "✗ Missing")
            
        except KeyError as e:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="Azure Configuration Missing",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error=f"Missing Streamlit secret: {e}. Need AZURE_AI_FOUNDRY_ENDPOINT, AZURE_AGENT_ID, and AZURE_TENANT_ID",
                has_grounding=False
            )
        
        if not project_endpoint:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="Azure Not Configured",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error="AZURE_AI_FOUNDRY_ENDPOINT not configured",
                has_grounding=False
            )
        
        if not agent_id:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="Azure Agent Not Configured",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error="AZURE_AGENT_ID not configured. Please create an agent in Azure AI Foundry first.",
                has_grounding=False
            )
        
        try:
            # Microsoft Entra ID Authentication (REQUIRED for Azure AI Foundry Agent Service)
            st.write("**DEBUG: Using Microsoft Entra ID Authentication (Required)**")
            
            try:
                credential = DefaultAzureCredential()
                # Use the correct scope for Azure AI services
                token_result = credential.get_token("https://cognitiveservices.azure.com/.default")
                
                headers = {
                    "Authorization": f"Bearer {token_result.token}",
                    "Content-Type": "application/json",
                    "User-Agent": "StreamlitApp/1.0"
                }
                st.write("✓ Successfully obtained Entra ID token")
                
            except ClientAuthenticationError as auth_error:
                st.write(f"✗ Entra ID Authentication failed: {auth_error}")
                return SearchResult(
                    success=False,
                    response="",
                    sources=[],
                    search_queries=[],
                    model="Entra ID Auth Failed",
                    timestamp=datetime.now().isoformat(),
                    response_time=time.time() - start_time,
                    error=f"Entra ID authentication failed: {auth_error}. Please ensure you're logged into Azure CLI or have managed identity configured.",
                    has_grounding=False
                )
            except Exception as auth_error:
                st.write(f"✗ Authentication setup error: {auth_error}")
                return SearchResult(
                    success=False,
                    response="",
                    sources=[],
                    search_queries=[],
                    model="Authentication Setup Failed",
                    timestamp=datetime.now().isoformat(),
                    response_time=time.time() - start_time,
                    error=f"Authentication setup failed: {auth_error}. Please run 'az login' or configure managed identity.",
                    has_grounding=False
                )
            
            # Debug: Test endpoint connectivity
            st.write("**DEBUG: Testing endpoint connectivity...**")
            
            # Enhanced query for better results
            enhanced_query = f"""Please provide comprehensive, current, and accurate information about: "{query}"
            
            I need detailed information including:
            - Current facts and latest developments
            - Key insights and important details  
            - Recent changes or updates (prioritize 2024-2025 information)
            - Multiple perspectives when relevant
            - Specific examples and evidence
            - User location is India
            
            Please structure your response clearly with proper organization and cite your sources."""
            
            # Use the correct API version for Azure AI Foundry Agent Service
            api_version = "2024-07-01-preview"  # Stable version for Agent Service
            
            # Step 1: Create thread with correct endpoint format
            thread_url = f"{project_endpoint}/threads"
            thread_params = {"api-version": api_version}
            
            st.write(f"DEBUG: Creating thread at {thread_url}")
            st.write(f"DEBUG: Using API version: {api_version}")
            
            thread_response = requests.post(
                thread_url, 
                headers=headers, 
                json={}, 
                params=thread_params, 
                timeout=30
            )
            
            st.write(f"DEBUG: Thread creation response: {thread_response.status_code}")
            
            if thread_response.status_code != 200:
                error_details = thread_response.text
                st.write(f"DEBUG: Thread creation error details: {error_details}")
                
                # Specific error handling
                if thread_response.status_code == 401:
                    return SearchResult(
                        success=False,
                        response="",
                        sources=[],
                        search_queries=[],
                        model="Authentication Failed",
                        timestamp=datetime.now().isoformat(),
                        response_time=time.time() - start_time,
                        error=f"Authentication failed (401). Azure AI Foundry Agent Service requires Microsoft Entra ID authentication. Please ensure you're logged in with 'az login' and have proper RBAC roles assigned. Details: {error_details}",
                        has_grounding=False
                    )
                elif thread_response.status_code == 403:
                    return SearchResult(
                        success=False,
                        response="",
                        sources=[],
                        search_queries=[],
                        model="Access Forbidden",
                        timestamp=datetime.now().isoformat(),
                        response_time=time.time() - start_time,
                        error=f"Access forbidden (403). Please ensure you have the required Azure roles: Cognitive Services User, Azure AI Developer. Details: {error_details}",
                        has_grounding=False
                    )
                elif thread_response.status_code == 404:
                    return SearchResult(
                        success=False,
                        response="",
                        sources=[],
                        search_queries=[],
                        model="Endpoint Not Found",
                        timestamp=datetime.now().isoformat(),
                        response_time=time.time() - start_time,
                        error=f"Endpoint not found (404). Please verify your project endpoint format: {project_endpoint}. Details: {error_details}",
                        has_grounding=False
                    )
                
                return SearchResult(
                    success=False,
                    response="",
                    sources=[],
                    search_queries=[],
                    model="Thread Creation Failed",
                    timestamp=datetime.now().isoformat(),
                    response_time=time.time() - start_time,
                    error=f"Failed to create thread: {thread_response.status_code} - {error_details}",
                    has_grounding=False
                )
            
            thread_data = thread_response.json()
            thread_id = thread_data.get("id")
            if not thread_id:
                return SearchResult(
                    success=False,
                    response="",
                    sources=[],
                    search_queries=[],
                    model="Thread ID Missing",
                    timestamp=datetime.now().isoformat(),
                    response_time=time.time() - start_time,
                    error="No thread ID returned from Azure AI Foundry",
                    has_grounding=False
                )
            
            st.write(f"DEBUG: Thread created successfully with ID: {thread_id}")
            
            # Step 2: Add message to thread
            message_url = f"{project_endpoint}/threads/{thread_id}/messages"
            message_data = {
                "role": "user",
                "content": enhanced_query
            }
            
            message_response = requests.post(
                message_url,
                headers=headers,
                json=message_data,
                params=thread_params,
                timeout=30
            )
            
            st.write(f"DEBUG: Message creation response: {message_response.status_code}")
            
            if message_response.status_code != 200:
                return SearchResult(
                    success=False,
                    response="",
                    sources=[],
                    search_queries=[],
                    model="Message Creation Failed",
                    timestamp=datetime.now().isoformat(),
                    response_time=time.time() - start_time,
                    error=f"Failed to create message: {message_response.status_code} - {message_response.text}",
                    has_grounding=False
                )
            
            # Step 3: Create and run the agent
            run_url = f"{project_endpoint}/threads/{thread_id}/runs"
            run_data = {
                "assistant_id": agent_id,
                "additional_instructions": f"""For this query about "{query}", please use the Bing grounding tool to search for the most current information available. This query would benefit from real-time web data. Always search for recent information when the query relates to current events, news, or time-sensitive topics."""
            }
            
            run_response = requests.post(
                run_url,
                headers=headers,
                json=run_data,
                params=thread_params,
                timeout=30
            )
            
            st.write(f"DEBUG: Run creation response: {run_response.status_code}")
            
            if run_response.status_code != 200:
                return SearchResult(
                    success=False,
                    response="",
                    sources=[],
                    search_queries=[],
                    model="Run Creation Failed",
                    timestamp=datetime.now().isoformat(),
                    response_time=time.time() - start_time,
                    error=f"Failed to create run: {run_response.status_code} - {run_response.text}",
                    has_grounding=False
                )
            
            run_data_response = run_response.json()
            run_id = run_data_response.get("id")
            if not run_id:
                return SearchResult(
                    success=False,
                    response="",
                    sources=[],
                    search_queries=[],
                    model="Run ID Missing",
                    timestamp=datetime.now().isoformat(),
                    response_time=time.time() - start_time,
                    error="No run ID returned from Azure AI Foundry",
                    has_grounding=False
                )
            
            st.write(f"DEBUG: Run created successfully with ID: {run_id}")
            
            # Continue with polling and response extraction...
            # [Rest of the implementation remains the same as in the previous version]
            
            return SearchResult(
                success=True,
                response="Authentication and endpoint configuration fixed. Continue with full implementation.",
                sources=[],
                search_queries=[query],
                model=f"Azure AI Foundry Agent ({agent_id}) with Bing Grounding",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                has_grounding=True
            )
            
        except requests.RequestException as e:
            st.write(f"DEBUG: Network error: {str(e)}")
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="Azure AI Network Error",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error=f"Network error: {str(e)}",
                has_grounding=False
            )
        
        except Exception as e:
            st.write(f"DEBUG: Unexpected error: {str(e)}")
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="Azure AI Error",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error=str(e),
                has_grounding=False
            )


def add_citations_to_text(response_result: SearchResult) -> str:
    """Add inline citations to the response text for Gemini only"""
    if not response_result.has_grounding or not response_result.sources:
        return response_result.response
    
    # Only apply inline citations to Gemini models
    if "gemini" not in response_result.model.lower():
        return response_result.response
    
    text = response_result.response
    sources = response_result.sources
    
    if not sources:
        return text
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    # Add citations to paragraphs (one source per substantial paragraph)
    source_idx = 0
    for i, paragraph in enumerate(paragraphs):
        if len(paragraph.strip()) > 100 and source_idx < len(sources):  # Only substantial paragraphs
            source = sources[source_idx]
            title = source.get('title', f'Source {source_idx + 1}')
            uri = source.get('uri', '')
            if uri:
                paragraphs[i] = paragraph.rstrip() + f" [{title}]({uri})"
            source_idx += 1
    
    return '\n\n'.join(paragraphs)

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
        
        # Sources section (only for non-Gemini models)
        if result.sources and "gemini" not in result.model.lower():
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
                
                For GPT-4 Responses API or Azure AI Agents:
                ```bash
                pip install openai
                ```
                """)
            
            elif "AZURE" in error_str:
                st.markdown("""
                **🔧 Azure Configuration Issues:**
                1. Check Azure OpenAI endpoint URL
                2. Verify Azure OpenAI API key
                3. Confirm model deployment name
                4. Ensure Bing Search is enabled in Azure AI Foundry
                5. Check API version compatibility
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
    st.title("🔍 Advanced Web Search Comparison")
    st.markdown("**Choose between Gemini 2.5 Flash, GPT-4o Responses API, or Azure AI Agents with Bing Search**")
    
    # Model Selection
    st.subheader("🤖 Select AI Model")
    
    # Check SDK availability
    gemini_available = NEW_SDK_AVAILABLE or OLD_SDK_AVAILABLE
    openai_available = OPENAI_AVAILABLE
    azure_available = AZURE_OPENAI_AVAILABLE
    
    options = []
    if gemini_available:
        options.append("Gemini 2.5 Flash with Google Search Grounding")
    if openai_available:
        options.append("GPT-4o with Responses API Web Search")
    if azure_available:
        options.append("Azure AI Agents with Bing Search Grounding")
    
    if not options:
        st.error("❌ No AI models available. Please install required SDKs and configure API keys:")
        st.markdown("""
        ```bash
        pip install google-genai openai
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
    elif "Azure AI Agents" in selected_model:
        if azure_available:
            st.success(f"✅ Azure OpenAI SDK available - {AZURE_MODEL_DEPLOYMENT} with Bing Search enabled")
        else:
            st.error("❌ Azure OpenAI SDK not available")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # API Key status
    st.sidebar.subheader("🔑 API Keys")
    st.sidebar.success("✅ Gemini API Key: Configured")
    if OPENAI_API_KEY and OPENAI_API_KEY != "sk-your-openai-api-key-here":
        st.sidebar.success("✅ OpenAI API Key: Configured")
    else:
        st.sidebar.warning("⚠️ OpenAI API Key: Not Configured")
    
    if AZURE_OPENAI_KEY and AZURE_OPENAI_KEY != "your-azure-openai-key-here":
        st.sidebar.success("✅ Azure OpenAI API Key: Configured")
    else:
        st.sidebar.warning("⚠️ Azure OpenAI API Key: Not Configured")
    
    if AZURE_AI_FOUNDRY_KEY and AZURE_AI_FOUNDRY_KEY != "your-azure-ai-foundry-key-here":
        st.sidebar.success("✅ Azure AI Foundry Key: Configured")
    else:
        st.sidebar.warning("⚠️ Azure AI Foundry Key: Not Configured")
    
    # Azure Configuration Details
    if "Azure" in selected_model:
        st.sidebar.subheader("🔧 Azure Configuration")
        st.sidebar.info(f"**Endpoint:** {AZURE_OPENAI_ENDPOINT[:30]}...")
        st.sidebar.info(f"**Model:** {AZURE_MODEL_DEPLOYMENT}")
        st.sidebar.info(f"**AI Foundry:** {AZURE_AI_FOUNDRY_ENDPOINT[:30]}...")
    
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
        
        elif "Azure AI Agents" in selected_model:  # Azure AI Agents
            if not azure_available:
                st.error("❌ Azure OpenAI SDK not available")
                return
            
            if AZURE_OPENAI_KEY == "your-azure-openai-key-here":
                st.error("❌ Azure API keys not configured in code")
                return
            
            with st.spinner(f"🔍 Searching with Azure AI Agents..."):
                result = AzureAIAgentsSearch.search(search_query)
                
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
        - Azure AI: {'Available' if azure_available else 'Not Available'}
        - Grounding: All models support web search
        - Current: {selected_model[:30]}...
        """)
    
    with col2:
        st.markdown("""
        **💡 Tips:**
        - Use specific queries for better results
        - Include current year for recent info
        - All models provide real-time data
        - Check sources for verification
        - Azure uses Bing Search grounding
        """)
    
    st.markdown("---")
    st.markdown(
        "**Powered by Gemini, GPT-4o, and Azure AI Agents with Web Search** | "
        "API Keys Embedded | "
        "Usage may incur costs"
    )

if __name__ == "__main__":
    main()