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
    except ImportError:
        OLD_SDK_AVAILABLE = False
        st.error("Neither new nor old Google GenAI SDK is available")

# OpenAI SDK
from openai import OpenAI

@dataclass
class SearchResult:
    service: str
    query: str
    response: str
    sources: List[Dict[str, str]]
    response_time: float
    has_grounding: bool
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

def search_gemini_new_sdk(query: str) -> SearchResult:
    """Search using new Google GenAI SDK with grounding"""
    start_time = time.time()
    sources = []
    has_grounding = False
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Create enhanced query with structured prompt
        optimized_query = f"""Please provide comprehensive, current, and accurate information about: "{query}"
I need detailed information including:
- Current facts and latest developments
- Key insights and important details
- Recent changes or updates (prioritize 2024/2025 information)
- Multiple perspectives when relevant
- Specific examples and evidence
- User location is India
Please structure your response clearly with proper organization and cite your sources."""
        
        # Configure model with grounding
        response = client.models.generate_content(
            model='gemini-1.5-pro',
            contents=optimized_query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                system_instruction="You are a helpful AI assistant that provides comprehensive, accurate, and well-sourced information. Always cite your sources when providing information."
            )
        )
        
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # Check for grounding information in the response
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                has_grounding = True
                grounding_metadata = candidate.grounding_metadata
                
                # Extract grounding chunks if available
                if hasattr(grounding_metadata, 'grounding_chunks'):
                    for chunk in grounding_metadata.grounding_chunks:
                        if hasattr(chunk, 'web') and chunk.web:
                            sources.append({
                                "title": chunk.web.title if hasattr(chunk.web, 'title') else "Web Source",
                                "uri": chunk.web.uri if hasattr(chunk.web, 'uri') else ""
                            })
        
        response_time = time.time() - start_time
        
        return SearchResult(
            service="Gemini (New SDK) with Grounding",
            query=query,
            response=response_text,
            sources=sources,
            response_time=response_time,
            has_grounding=has_grounding
        )
    
    except Exception as e:
        response_time = time.time() - start_time
        return SearchResult(
            service="Gemini (New SDK) with Grounding",
            query=query,
            response="",
            sources=[],
            response_time=response_time,
            has_grounding=False,
            error=str(e)
        )

def search_gemini_legacy_sdk(query: str) -> SearchResult:
    """Search using legacy Google GenAI SDK"""
    start_time = time.time()
    sources = []
    has_grounding = False
    
    try:
        genai_old.configure(api_key=GEMINI_API_KEY)
        model = genai_old.GenerativeModel('gemini-1.5-pro')
        
        # Create enhanced query with structured prompt
        prompt = f"""Please provide comprehensive, current, and accurate information about: "{query}"
I need detailed information including:
- Current facts and latest developments
- Key insights and important details
- Recent changes or updates (prioritize 2024/2025 information)
- Multiple perspectives when relevant
- Specific examples and evidence
- User location is India
Please structure your response clearly with proper organization and cite your sources."""
        
        response = model.generate_content(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        response_time = time.time() - start_time
        
        return SearchResult(
            service="Gemini (Legacy SDK)",
            query=query,
            response=response_text,
            sources=sources,
            response_time=response_time,
            has_grounding=has_grounding
        )
    
    except Exception as e:
        response_time = time.time() - start_time
        return SearchResult(
            service="Gemini (Legacy SDK)",
            query=query,
            response="",
            sources=[],
            response_time=response_time,
            has_grounding=False,
            error=str(e)
        )

def search_gpt4_responses_api(query: str) -> SearchResult:
    """Search using OpenAI GPT-4 with web search capability"""
    start_time = time.time()
    sources = []
    has_grounding = False
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Enhanced query with comprehensive instructions
        enhanced_query = f"""Please provide comprehensive, current, and accurate information about: "{query}"
I need detailed information including:
- Current facts and latest developments
- Key insights and important details
- Recent changes or updates (prioritize 2024/2025 information)
- Multiple perspectives when relevant
- Specific examples and evidence
- User location is India
Please structure your response clearly with proper organization and cite your sources."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": enhanced_query}],
            temperature=0.3,
            max_tokens=1500,
            tools=[
                {
                    "type": "web_search",
                    "web_search": {
                        "max_results": 10
                    }
                }
            ]
        )
        
        response_content = response.choices[0].message.content
        
        # Check for tool calls (web search results)
        if response.choices[0].message.tool_calls:
            has_grounding = True
            for tool_call in response.choices[0].message.tool_calls:
                if tool_call.type == "web_search":
                    # Extract search results from tool call
                    search_results = tool_call.web_search.results if hasattr(tool_call, 'web_search') else []
                    for result in search_results:
                        sources.append({
                            "title": result.get("title", "Web Search Result"),
                            "uri": result.get("url", "")
                        })
        
        response_time = time.time() - start_time
        
        return SearchResult(
            service="GPT-4 Responses API with Web Search",
            query=query,
            response=response_content,
            sources=sources,
            response_time=response_time,
            has_grounding=has_grounding
        )
    
    except Exception as e:
        response_time = time.time() - start_time
        return SearchResult(
            service="GPT-4 Responses API with Web Search",
            query=query,
            response="",
            sources=[],
            response_time=response_time,
            has_grounding=False,
            error=str(e)
        )

def search_azure_ai_agents(query: str) -> SearchResult:
    """Search using Azure AI Agents with Bing Search grounding"""
    start_time = time.time()
    sources = []
    has_grounding = False
    
    try:
        # Step 1: Create a thread
        thread_url = f"{AZURE_AI_FOUNDRY_ENDPOINT}/openai/threads"
        headers = {
            "api-key": AZURE_AI_FOUNDRY_KEY,
            "Content-Type": "application/json"
        }
        
        thread_response = requests.post(thread_url, headers=headers, json={})
        if thread_response.status_code != 200:
            raise Exception(f"Failed to create thread: {thread_response.text}")
        
        thread_id = thread_response.json()["id"]
        
        # Step 2: Add message to thread with enhanced query
        enhanced_query = f"""Please provide comprehensive, current, and accurate information about: "{query}"
I need detailed information including:
- Current facts and latest developments
- Key insights and important details
- Recent changes or updates (prioritize 2024/2025 information)
- Multiple perspectives when relevant
- Specific examples and evidence
- User location is India
Please structure your response clearly with proper organization and cite your sources."""
        
        message_url = f"{AZURE_AI_FOUNDRY_ENDPOINT}/openai/threads/{thread_id}/messages"
        message_data = {
            "role": "user",
            "content": enhanced_query
        }
        
        message_response = requests.post(message_url, headers=headers, json=message_data)
        if message_response.status_code != 200:
            raise Exception(f"Failed to create message: {message_response.text}")
        
        # Step 3: Run the agent
        run_url = f"{AZURE_AI_FOUNDRY_ENDPOINT}/openai/threads/{thread_id}/runs"
        run_data = {
            "assistant_id": AZURE_AGENT_ID
        }
        
        run_response = requests.post(run_url, headers=headers, json=run_data)
        if run_response.status_code != 200:
            raise Exception(f"Failed to create run: {run_response.text}")
        
        run_id = run_response.json()["id"]
        
        # Step 4: Poll for completion
        status_url = f"{AZURE_AI_FOUNDRY_ENDPOINT}/openai/threads/{thread_id}/runs/{run_id}"
        
        max_wait_time = 60  # 60 seconds timeout
        start_poll_time = time.time()
        
        while time.time() - start_poll_time < max_wait_time:
            status_response = requests.get(status_url, headers=headers)
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data.get("status")
                
                if status == "completed":
                    break
                elif status in ["failed", "cancelled", "expired"]:
                    raise Exception(f"Run failed with status: {status}")
            
            time.sleep(2)  # Wait 2 seconds between polls
        else:
            raise Exception("Timeout waiting for agent response")
        
        # Step 5: Get the messages
        messages_url = f"{AZURE_AI_FOUNDRY_ENDPOINT}/openai/threads/{thread_id}/messages"
        messages_response = requests.get(messages_url, headers=headers)
        
        if messages_response.status_code != 200:
            raise Exception(f"Failed to get messages: {messages_response.text}")
        
        messages_data = messages_response.json()
        
        # Extract the assistant's response
        response_content = ""
        for message in messages_data.get("data", []):
            if message.get("role") == "assistant":
                content = message.get("content", [])
                for content_part in content:
                    if content_part.get("type") == "text":
                        text_content = content_part.get("text", {})
                        response_content += text_content.get("value", "")
                        
                        # Extract annotations (citations/sources)
                        annotations = text_content.get("annotations", [])
                        for annotation in annotations:
                            annotation_type = annotation.get("type")
                            if annotation_type == "file_citation":
                                file_citation = annotation.get("file_citation", {})
                                sources.append({
                                    "title": f"File: {file_citation.get('file_id', 'Unknown')}",
                                    "uri": f"file://{file_citation.get('file_id', '')}"
                                })
                                has_grounding = True
                            elif annotation_type == "file_path":
                                file_path = annotation.get("file_path", {})
                                sources.append({
                                    "title": f"File: {file_path.get('file_id', 'Unknown')}",
                                    "uri": f"file://{file_path.get('file_id', '')}"
                                })
                                has_grounding = True
                            elif annotation_type == "url_citation":
                                # ADDED: Handle URL citations from Bing search
                                url_citation = annotation.get("url_citation", {})
                                if url_citation:
                                    title = url_citation.get("title", url_citation.get("name", "Unknown Source"))
                                    url = url_citation.get("url", url_citation.get("uri", ""))
                                    if url:  # Only add if URL exists
                                        sources.append({
                                            "title": title,
                                            "uri": url
                                        })
                                        has_grounding = True
                        break
                break
        
        response_time = time.time() - start_time
        
        return SearchResult(
            service="Azure AI Agents with Bing Search",
            query=query,
            response=response_content,
            sources=sources,
            response_time=response_time,
            has_grounding=has_grounding
        )
    
    except Exception as e:
        response_time = time.time() - start_time
        return SearchResult(
            service="Azure AI Agents with Bing Search",
            query=query,
            response="",
            sources=[],
            response_time=response_time,
            has_grounding=False,
            error=str(e)
        )

def display_search_result(result: SearchResult):
    """Display a single search result"""
    with st.container():
        # Header with service name and grounding indicator
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            grounding_icon = "🔗" if result.has_grounding else "❌"
            st.subheader(f"{grounding_icon} {result.service}")
        with col2:
            st.metric("Response Time", f"{result.response_time:.2f}s")
        with col3:
            st.metric("Sources", len(result.sources))
        
        # Display error if present
        if result.error:
            st.error(f"Error: {result.error}")
            return
        
        # Display response
        if result.response:
            st.markdown("**Response:**")
            st.markdown(result.response)
        
        # Display sources if available
        if result.sources:
            st.markdown("**Sources:**")
            for i, source in enumerate(result.sources, 1):
                title = source.get("title", "Unknown Title")
                uri = source.get("uri", "")
                if uri:
                    st.markdown(f"{i}. [{title}]({uri})")
                else:
                    st.markdown(f"{i}. {title}")
        
        st.divider()

def main():
    st.set_page_config(
        page_title="AI Web Search Comparison",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 AI Web Search API Comparison")
    st.markdown("Compare search results across different AI services with web grounding capabilities")
    
    # Sidebar for service selection
    with st.sidebar:
        st.header("Service Configuration")
        
        # Check service availability
        services_available = {
            "Gemini (New SDK) with Grounding": NEW_SDK_AVAILABLE,
            "Gemini (Legacy SDK)": OLD_SDK_AVAILABLE,
            "GPT-4 Responses API": True,  # Assuming OpenAI SDK is available
            "Azure AI Agents with Bing": True   # Assuming requests is available
        }
        
        # Display service status
        st.markdown("**Service Status:**")
        for service, available in services_available.items():
            icon = "✅" if available else "❌"
            st.markdown(f"{icon} {service}")
        
        st.divider()
        
        # Service selection
        st.markdown("**Select Services to Test:**")
        selected_services = []
        
        if services_available["Gemini (New SDK) with Grounding"]:
            if st.checkbox("Gemini (New SDK) with Grounding", value=True):
                selected_services.append("gemini_new")
        
        if services_available["Gemini (Legacy SDK)"]:
            if st.checkbox("Gemini (Legacy SDK)", value=True):
                selected_services.append("gemini_legacy")
        
        if services_available["GPT-4 Responses API"]:
            if st.checkbox("GPT-4 Responses API", value=True):
                selected_services.append("gpt4_responses")
        
        if services_available["Azure AI Agents with Bing"]:
            if st.checkbox("Azure AI Agents with Bing Search", value=True):
                selected_services.append("azure_agents")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., Latest developments in AI technology 2024",
            help="Enter a question or topic you want to search for"
        )
    
    with col2:
        st.markdown("&nbsp;")  # Spacing
        search_button = st.button("🔍 Search", type="primary")
    
    # Search execution
    if search_button and query:
        if not selected_services:
            st.warning("Please select at least one service to test.")
            return
        
        st.markdown("---")
        st.markdown(f"### Search Results for: *{query}*")
        
        # Track results for comparison
        results = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_services = len(selected_services)
        
        for i, service in enumerate(selected_services):
            status_text.text(f"Searching with {service}...")
            progress_bar.progress((i) / total_services)
            
            try:
                if service == "gemini_new":
                    result = search_gemini_new_sdk(query)
                elif service == "gemini_legacy":
                    result = search_gemini_legacy_sdk(query)
                elif service == "gpt4_responses":
                    result = search_gpt4_responses_api(query)
                elif service == "azure_agents":
                    result = search_azure_ai_agents(query)
                
                results.append(result)
                display_search_result(result)
                
            except Exception as e:
                st.error(f"Error with {service}: {str(e)}")
        
        progress_bar.progress(1.0)
        status_text.text("Search completed!")
        
        # Summary comparison
        if len(results) > 1:
            st.markdown("### 📊 Summary Comparison")
            
            comparison_data = []
            for result in results:
                comparison_data.append({
                    "Service": result.service,
                    "Response Time": f"{result.response_time:.2f}s",
                    "Has Grounding": "Yes" if result.has_grounding else "No",
                    "Number of Sources": len(result.sources),
                    "Status": "Success" if not result.error else "Error"
                })
            
            st.dataframe(comparison_data, use_container_width=True)
            
            # Performance insights
            fastest_service = min(results, key=lambda x: x.response_time)
            most_sources = max(results, key=lambda x: len(x.sources))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fastest Service", fastest_service.service, f"{fastest_service.response_time:.2f}s")
            with col2:
                st.metric("Most Sources", most_sources.service, f"{len(most_sources.sources)} sources")
    
    elif search_button and not query:
        st.warning("Please enter a search query.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <small>AI Web Search Comparison Tool | Compare multiple AI search services side by side</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
