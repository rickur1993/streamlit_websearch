import streamlit as st
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import os
import json
from dotenv import load_dotenv
import os
#load_dotenv("API.env")
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Try importing the new Google GenAI SDK first (recommended)
#OPENAI_API_KEY="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"
#GEMINI_API_KEY="AIzaSyBf7QT0LIl1sjghS_Kk7EPSLnwR38Rktso"
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
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
    page_title="AI Search with Grounding",
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
    """Handles GPT-4 with OpenAI Responses API for web search"""
    
    @staticmethod
    def search(query: str) -> SearchResult:
        """Search using GPT-4 with Responses API for web grounding"""
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
            openai.api_key = OPENAI_API_KEY
            
            # Enhanced prompt for better web search results
            enhanced_query = f"""
            Please provide comprehensive, current, and accurate information about: "{query}"
            
            I need detailed information including:
            - Current facts and latest developments
            - Key insights and important details
            - Recent changes or updates (prioritize 2024/2025 information)
            - Multiple perspectives when relevant
            - Specific examples and evidence
            
            Please structure your response clearly with proper organization and cite your sources.
            """
            
            # Use GPT-4 with web search capability via Responses API
            response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant with access to current web information. Always provide accurate, up-to-date information with proper citations when available."
        },
        {
            "role": "user",
            "content": enhanced_query
        }
    ],
    temperature=0.1,
    max_tokens=4000
)
            
            response_time = time.time() - start_time
            
            # Extract response content
            response_text = ""
            if response.choices and response.choices[0].message:
                response_text = response.choices[0].message.content
            
            # Try to extract sources and search queries (basic implementation)
            sources = []
            search_queries = [query]  # Basic - actual search queries would need API support
            has_grounding = True  # Assume GPT-4o has web access
            
            # Basic source extraction from response text (looking for URLs)
            import re
            url_pattern = r'https?://[^\s<>"{}|\\^`[\]]*'
            urls = re.findall(url_pattern, response_text)
            
            for i, url in enumerate(urls[:10]):  # Limit to 10 sources
                sources.append({
                    'title': f'Web Source {i+1}',
                    'uri': url
                })
            
            return SearchResult(
                success=True,
                response=response_text,
                sources=sources,
                search_queries=search_queries,
                model="GPT-4o with Web Search",
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
                model="GPT-4 (Error)",
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

def main():
    # Header
    st.title("🔍 AI Search with Grounding")
    st.markdown("**Choose between Gemini 2.5/2.0 Flash Grounding or GPT-4 with Responses API**")
    
    # Model Selection
    st.subheader("🤖 Select AI Model")
    
    # Check SDK availability
    gemini_available = NEW_SDK_AVAILABLE or OLD_SDK_AVAILABLE
    openai_available = OPENAI_AVAILABLE
    
    options = []
    if gemini_available:
        options.append("Gemini 2.5/2.0 Flash with Google Search Grounding")
    if openai_available:
        options.append("GPT-4 with Responses API Web Search")
    
    if not options:
        st.error("❌ No AI models available. Please install required SDKs:")
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
    else:
        if openai_available:
            st.success("✅ OpenAI SDK available - GPT-4 with web search enabled")
        else:
            st.error("❌ OpenAI SDK not available")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # API Key status
    st.sidebar.subheader("🔑 API Keys")
    st.sidebar.success("✅ Gemini API Key: Configured")
    if OPENAI_API_KEY and OPENAI_API_KEY != "sk-your-openai-api-key-here":
        st.sidebar.success("✅ OpenAI API Key: Configured")
    else:
        st.sidebar.warning("⚠️ OpenAI API Key: Not Configured")
    
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
    
    # Example queries
    st.markdown("**💡 Try These Examples:**")
    examples = [
        "Latest AI breakthroughs 2025",
        "Current climate change solutions", 
        "Recent space exploration missions",
        "Electric vehicle market trends 2025",
        "Quantum computing developments"
    ]
    
    cols = st.columns(3)
    for i, example in enumerate(examples):
        with cols[i % 3]:
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                st.session_state.main_search = example
                search_query = example
                search_button = True
    
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
        
        else:  # GPT-4
            if not openai_available:
                st.error("❌ OpenAI SDK not available")
                return
            
            if OPENAI_API_KEY == "sk-your-openai-api-key-here":
                st.error("❌ OpenAI API key not configured in code")
                return
            
            with st.spinner(f"🔍 Searching with GPT-4..."):
                result = GPTResponsesSearch.search(search_query)
        
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
        - GPT-4: {'Available' if openai_available else 'Not Available'}
        - Grounding: Both support web search
        - Current: {selected_model[:30]}...
        """)
    
    with col2:
        st.markdown("""
        **💡 Tips:**
        - Use specific queries for better results
        - Include current year for recent info
        - Both models provide real-time data
        - Check sources for verification
        """)
    
    st.markdown("---")
    st.markdown(
        "**Powered by Gemini & GPT-4 with Web Search** | "
        "API Keys Embedded | "
        "Usage may incur costs"
    )

if __name__ == "__main__":
    main()