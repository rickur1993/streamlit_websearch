# file: GEMINI_GPT_V1.py

import os
import re
import time
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
import streamlit as st

# Configure API key: recommended to set GEMINI_API_KEY in the environment
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

class GeminiGroundingSearch:
    """
    Grounded search with:
    - True streaming output (no "streaming" label shown)
    - Inline citations as shortened clickable URL links
    - No separate sources or "type of searches" sections rendered
    - Adaptive prompt (omits irrelevant sections)
    - Immediate completion after streaming finishes
    - Follow-up question suggestions
    """

    def __init__(self, api_key: str):
        try:
            from google import genai
            from google.genai import types
            os.environ["GEMINI_API_KEY"] = api_key
            self.client = genai.Client()
            self.types = types
        except ImportError:
            st.error("Google GenAI SDK not found. Install: pip install google-genai")
            raise

    def _adaptive_prompt(self, query: str) -> str:
        """
        Generic adaptive prompt with optional finance-specific enhancements.
        """
        
        # Detect query characteristics
        is_comparative = any(kw in query.lower() for kw in 
                            ['vs', 'versus', 'compare', 'comparison', 'between', 'difference'])
        is_financial = any(kw in query.lower() for kw in 
                        ['market', 'stock', 'financial', 'revenue', 'earnings', 'valuation', 
                        'investment', 'portfolio', 'budget', 'forecast', 'analysis', 'price'])
        
        return f"""
    Analyze the following query and provide a comprehensive, well-structured response using up-to-date grounded information.

    Query: {query}

    RESPONSE STRUCTURE:
    ## Executive Summary (for formal/business queries) OR ## Overview (for general queries)
    - 2-3 sentence direct answer addressing the core question
    - Include key quantitative details (numbers, dates, percentages) when relevant

    ## Main Analysis
    - Present comprehensive analysis organized by logical themes
    - Support claims with specific data, facts, and recent developments
    - Maintain consistent depth across all evaluated aspects

    ADAPTIVE CONTENT SECTIONS (include only when directly relevant):
    - **Key Facts & Figures**: Quantitative data, metrics, statistics with dates/sources
    - **Timeline/Historical Context**: Chronological developments, milestones, evolution
    - **Current Status**: Latest updates, present state (with specific dates)
    - **Comparative Analysis**: Side-by-side evaluation using structured tables
    - **Technical Details**: Specifications, methodologies, implementation details
    - **Impact Analysis**: Implications, consequences, affected parties
    - **Regional/Geographic Breakdown**: Location-specific information
    - **Risk Assessment**: Uncertainties, challenges, vulnerabilities
    - **Future Outlook**: Projections, trends, forward-looking analysis (omit for historical/concluded events)
    - **Recommendations**: Actionable guidance for decision-oriented queries (omit for informational/entertainment)
    

    {self._get_comparative_instructions() if is_comparative else ""}

    {self._get_financial_guidelines() if is_financial else ""}

    QUERY-TYPE ADAPTATIONS:
    - **Factual/Informational** → Concise facts, dates, numbers with authoritative sources
    - **Comparative** → Structured tables with identical parameters for all entities
    - **How-to/Technical** → Step-by-step instructions, clear action items
    - **Current Events** → Timeline + current status + recent developments + implications
    - **Historical** → Chronological narrative + impact (no future predictions)
    - **Sports/Entertainment** → Results, highlights, performance (no recommendations)
    - **Financial/Market** → Metrics-driven with tables, specific figures, dates, and context
    
    Generic guilines for all queries:
    -** Focus on Indian context where applicable**
    - Use recent, authoritative sources (preferably within last 6 months)
    - Always prioritize recent events or ongoing events in case of repeatative events
    - Avoid vague language; be specific and precise
    - Do NOT fabricate or assume data; use "Not available" if unknown
    - Maintain neutral, objective tone; avoid speculation
    - Prioritize clarity and logical flow; use headings, bullet points, tables
    - Ensure proper grammar, spelling, and punctuation

    FORMATTING STANDARDS:
    - Use ## for main sections, ### for subsections
    - **Bold** key terms, metrics, and entity names (max 3 consecutive words per bold)
    - Use markdown tables for all comparative data and multi-dimensional information
    - Use bullet points (- or *) for lists; never nest lists
    - DO NOT include citation numbers like [1], [2] - citations will be added automatically
    - Ensure complete response without truncation
    

    CONSISTENCY PRINCIPLES:
    - Apply systematic methodology to ensure comprehensive coverage
    - Use identical evaluation criteria across all compared entities
    - Maintain deterministic structure for similar query types
    - Prioritize factual accuracy over creative variation
    """.strip()

    def _get_comparative_instructions(self) -> str:
        """Returns detailed instructions for comparative queries."""
        return """
    COMPARATIVE ANALYSIS PROTOCOL (MANDATORY FOR COMPARISON QUERIES):
    1. **Entity Identification**: List all entities being compared at the start
    2. **Parameter Definition**: Establish a FIXED set of evaluation dimensions applicable to ALL entities
    3. **Symmetric Evaluation**: Assess EVERY parameter for EVERY entity
    - Use "Not available" or "N/A" for missing data
    - Do NOT introduce entity-specific parameters that exclude others
    4. **Tabular Presentation**: Present as markdown table with:
    - Column 1: Parameter/Dimension name
    - Remaining columns: Each entity being compared
    - All table cells must be populated (no empty cells)
    5. **Measurement Consistency**: Use identical units, timeframes, and methodologies across entities

    Example comparison table structure:
    | Parameter | Entity A | Entity B | Entity C |
    |-----------|----------|----------|----------|
    | Key Metric 1 | Value | Value | Value |
    | Key Metric 2 | Value | Value | Value |

    """

    def _get_financial_guidelines(self) -> str:
        """Returns enhanced guidelines for financial queries."""
        return """
    FINANCIAL/MARKET ANALYSIS ENHANCEMENTS:
    - **Data Precision**: Always specify currency, time period, and fiscal year
    - **Comparative Metrics**: Include both absolute values AND percentage changes (YoY, QoQ, MoM)
    - **Context & Benchmarks**: Provide industry averages, peer comparisons, historical context
    - **Source Attribution**: Note data sources for financial figures (e.g., quarterly reports, analyst estimates)
    - **Temporal Specificity**: Include exact dates for time-sensitive market data
    - **Financial Ratios**: Where relevant, include standard metrics (P/E, ROE, debt-to-equity, margins, etc.)
    - **Market Conditions**: Note relevant macroeconomic factors influencing the metrics
    - **Distinguish**: Clearly differentiate between reported vs. adjusted/normalized figures

    """

    


    def _shorten_url_domain(self, url: str) -> str:
        """
        Extract and shorten URL to display format like 'example.com'
        """
        try:
            # Skip vertex AI redirect URLs
            if not url or "vertexaisearch.cloud.google.com" in url:
                return None
            
            # Handle URLs without protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            parsed = urlparse(url)
            netloc = parsed.netloc
            
            if not netloc:
                return None
            
            # Remove 'www.' prefix if present
            if netloc.startswith('www.'):
                netloc = netloc[4:]
            
            # Keep just domain.tld for cleaner display
            parts = netloc.split('.')
            if len(parts) >= 2:
                netloc = '.'.join(parts[-2:])
            
            return netloc if netloc else None
        except Exception as e:
            st.sidebar.warning(f"URL parsing error: {e}")
            return None

    def _generate_followup_questions(self, query: str, response: str) -> List[str]:
        """
        Generate 3 follow-up questions based on the original query and response.
        Uses a lightweight, fast model for quick generation.
        """
        try:
            prompt = f"""Based on this user query and the response provided, suggest 3 brief, natural follow-up questions the user might want to ask next.

Original Query: {query}

Response Summary: {response[:500]}...

Generate 3 concise follow-up questions (each under 15 words) that would naturally extend this conversation. Return ONLY the questions, one per line, without numbers or bullets.
"""
            
            # Use flash model for fast follow-up generation
            config = self.types.GenerateContentConfig(
                max_output_tokens=200,
                temperature=0.8,
            )
            
            result = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",  # Fastest model for quick suggestions
                contents=prompt,
                config=config,
            )
            
            if hasattr(result, 'text') and result.text:
                questions = [q.strip() for q in result.text.strip().split('\n') if q.strip()]
                # Remove any numbering or bullets
                questions = [re.sub(r'^[\d\.\-\*\)]+\s*', '', q) for q in questions]
                return questions[:3]
            
            return []
        except Exception as e:
            return []

    def search_with_grounding_stream(
        self,
        query: str,
        model: str = "gemini-2.0-flash-exp",
        max_tokens: int = 30000,
        temperature: float = 0.2,
        container=None,
        time_badge=None,
        status_badge=None,
    ) -> Dict[str, Any]:
        """
        Streams the model output live to the container without any "streaming" label.
        Extracts grounding metadata from streaming chunks and inserts clickable shortened URL links.
        Completes immediately after streaming finishes.
        Returns a dict with success, response, metadata, model_used, search_time, followup_questions.
        """
        start_time = time.time()
        try:
            grounding_tool = self.types.Tool(google_search=self.types.GoogleSearch())
            config = self.types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                tools=[grounding_tool]
            )

            enhanced_query = self._adaptive_prompt(query)
            full_text = ""
            grounding_meta = {"grounding_chunks": [], "grounding_supports": []}
            
            if container:
                placeholder = container.empty()

            # Streaming call
            stream = self.client.models.generate_content_stream(
                model=model,
                contents=enhanced_query,
                config=config,
            )

            # Stream chunks and extract metadata in real-time
            for chunk in stream:
                if hasattr(chunk, "text") and chunk.text:
                    full_text += chunk.text
                    if container:
                        placeholder.markdown(full_text)
                
                # Extract grounding metadata from each chunk
                chunk_meta = self._extract_grounding_metadata_from_chunk(chunk)
                if chunk_meta:
                    # Merge chunks
                    for ch in chunk_meta.get("grounding_chunks", []):
                        if ch not in grounding_meta["grounding_chunks"]:
                            grounding_meta["grounding_chunks"].append(ch)
                    # Merge supports
                    for sup in chunk_meta.get("grounding_supports", []):
                        if sup not in grounding_meta["grounding_supports"]:
                            grounding_meta["grounding_supports"].append(sup)
                
                # Update time badge unobtrusively
                if time_badge:
                    elapsed = time.time() - start_time
                    time_badge.metric("Time", f"{elapsed:.2f}s")
            #if grounding_meta["grounding_chunks"]:
                #st.sidebar.success(f"✓ {len(grounding_meta['grounding_chunks'])} sources found")
                
                # DEBUG: Show full structure of first chunk
                #st.sidebar.write("**First chunk full structure:**")
                # if grounding_meta['grounding_chunks']:
                #     import json
                #     first_chunk = grounding_meta['grounding_chunks'][0]
                #     #st.sidebar.json(first_chunk)  # This will show all available fields
                    
                #     # Also show all keys available in the chunk
                #     #st.sidebar.write("**Available keys in chunk:**")
                #     #st.sidebar.write(list(first_chunk.keys()))
                # else:
                #     st.sidebar.warning("⚠  No grounding sources detected")
            # Insert inline citations clickable shortened URL links immediately after streaming
            
            final_text = self._insert_inline_url_citations(full_text, grounding_meta)
            
            # Replace streamed text with inline-cited finalized text
            if container:
                placeholder.markdown(final_text, unsafe_allow_html=True)

            end_time = time.time()
            
            # Final time update
            if time_badge:
                time_badge.metric("Time", f"{(end_time - start_time):.2f}s")
            
            # Set status to Complete immediately
            if status_badge:
                status_badge.metric("Status", "Complete")

            # Generate follow-up questions
            followup_questions = self._generate_followup_questions(query, final_text)

            return {
                "success": True,
                "response": final_text,
                "grounding_metadata": grounding_meta,
                "model_used": model,
                "search_time": end_time - start_time,
                "followup_questions": followup_questions,
            }

        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            
            # Check if it's a quota error and provide helpful message
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if container:
                    container.error(
                        "⚠️ **Quota Exceeded**: You've hit the API rate limit. "
                        "Try using Flash model or wait a few minutes before trying again. "
                        f"\n\nDetails: {error_msg[:200]}"
                    )
            else:
                if container:
                    container.error(f"Error: {error_msg}")
            
            if time_badge:
                time_badge.metric("Time", f"{(end_time - start_time):.2f}s")
            if status_badge:
                status_badge.metric("Status", "Error")
            return {
                "success": False,
                "error": error_msg,
                "error_type": type(e).__name__,
                "response": None,
                "grounding_metadata": None,
                "model_used": model,
                "search_time": end_time - start_time,
                "followup_questions": [],
            }

    def _extract_grounding_metadata_from_chunk(self, chunk) -> Optional[Dict[str, Any]]:
        """
        Extract grounding metadata structure from a streaming chunk.
        Extracts both redirect URLs and attempts to get real URLs.
        """
        try:
            if hasattr(chunk, "candidates") and chunk.candidates:
                cand = chunk.candidates[0]
                if hasattr(cand, "grounding_metadata") and cand.grounding_metadata:
                    gm = cand.grounding_metadata
                    meta = {
                        "grounding_chunks": [],
                        "grounding_supports": [],
                    }

                    if getattr(gm, "grounding_chunks", None):
                        chunks = []
                        for ch in gm.grounding_chunks:
                            chunk_data = {}
                            
                            # Try to get web info
                            if hasattr(ch, "web") and ch.web:
                                # Get the URI (might be redirect)
                                uri = getattr(ch.web, "uri", "")
                                title = getattr(ch.web, "title", "")
                                
                                # Try to extract real URL from title or other fields
                                # Sometimes the real URL is in the title or we need to follow redirect
                                chunk_data = {
                                    "uri": uri,
                                    "title": title,
                                }
                                
                                # Check if there are other attributes that might have the real URL
                                if hasattr(ch.web, '__dict__'):
                                    for key, value in ch.web.__dict__.items():
                                        if key not in ['uri', 'title'] and value:
                                            chunk_data[key] = value
                            
                            # Also try retrievedContext if available (alternative source info)
                            elif hasattr(ch, "retrieved_context") and ch.retrieved_context:
                                if hasattr(ch.retrieved_context, "uri"):
                                    chunk_data = {
                                        "uri": ch.retrieved_context.uri,
                                        "title": getattr(ch.retrieved_context, "title", ""),
                                    }
                            
                            if chunk_data:
                                chunks.append(chunk_data)
                        
                        meta["grounding_chunks"] = chunks

                    if getattr(gm, "grounding_supports", None):
                        supports = []
                        for s in gm.grounding_supports:
                            if hasattr(s, "segment"):
                                supports.append({
                                    "text": getattr(s.segment, "text", ""),
                                    "start_index": getattr(s.segment, "start_index", 0),
                                    "end_index": getattr(s.segment, "end_index", 0),
                                    "grounding_chunk_indices": getattr(s, "grounding_chunk_indices", []),
                                })
                        meta["grounding_supports"] = supports

                    # DEBUG: Log what we extracted
                    #st.sidebar.write(f"Extracted {len(meta['grounding_chunks'])} chunks from this stream chunk")
                    if meta['grounding_chunks']:
                        first_chunk = meta['grounding_chunks'][0]
                        #st.sidebar.write(f"First chunk keys: {list(first_chunk.keys())}")
                        #st.sidebar.write(f"First chunk data: {str(first_chunk)[:200]}")

                    return meta
            return None
        except Exception as e:
            st.sidebar.error(f"Metadata extraction error: {e}")
            return None

    def _insert_inline_url_citations(self, text: str, gm: Optional[Dict[str, Any]]) -> str:
        """
        Insert inline citations as clickable shortened URL links using title field.
        The title field contains the actual domain (e.g., 'wikipedia.org').
        """
        if not gm or not gm.get("grounding_chunks"):
            return self._clean_redirects(text)

        chunks: List[Dict[str, str]] = gm.get("grounding_chunks", [])
        supports: List[Dict[str, Any]] = gm.get("grounding_supports", [])
        #st.sidebar.markdown("### 🐛 Citation Debug Info")
        #chunks: List[Dict[str, str]] = gm.get("grounding_chunks", [])
        #supports: List[Dict[str, Any]] = gm.get("grounding_supports", [])
        
        #st.sidebar.write(f"**Total chunks available:** {len(chunks)}")
        #st.sidebar.write(f"**Total supports available:** {len(supports)}")
        
        # Show sample chunk structure
        # if chunks:
        #     st.sidebar.write("**First chunk structure:**")
        #     st.sidebar.json(chunks[0])
        
        # # Show sample support structure
        # if supports:
        #     st.sidebar.write("**First support structure:**")
        #     st.sidebar.json(supports[0])
        cited_text = self._clean_redirects(text)
        cited_segments = set()
        successfully_cited = False

        # Primary approach: Match and cite specific segments
        for s in supports:
            segment_text = s.get("text", "").strip()
            if not segment_text or segment_text in cited_segments:
                continue

            chunk_indices = s.get("grounding_chunk_indices", [])
            if not chunk_indices:
                continue
            #st.sidebar.markdown(f"**Attempting to cite segment:** `{segment_text[:50]}...`")
            #st.sidebar.write(f"- Chunk indices: {chunk_indices}")
            # Create citation with clickable links using redirect URL but displaying title
            citation_parts = []
            for idx in chunk_indices:
                if idx < len(chunks):
                    url = chunks[idx].get("uri", "")  # This is the redirect URL
                    title = chunks[idx].get("title", "")  # This is the actual domain!
                    #st.sidebar.write(f"  - Chunk {idx}: URL={url[:40]}..., Title={title}")
                    if not title or not url:
                        continue

                    # Create HTML link: use redirect URL as href, but display the title (domain)
                    citation_parts.append(
                        f'<a href="{url}" target="_blank" '
                        f'style="color: #1a73e8; text-decoration: none; font-size: 0.9em;">'
                        f'{title}</a>'
                    )
            
            if not citation_parts:
                continue

            # Join multiple sources with commas in parentheses
            citation = " (" + ", ".join(citation_parts) + ")"

            # Find first occurrence of segment not already cited
            seg_escaped = re.escape(segment_text)
            pattern = re.compile(rf"({seg_escaped})(?!\s*[\(\<a])", flags=re.IGNORECASE)
            new_text, n_subs = pattern.subn(rf"\1{citation}", cited_text, count=1)
            #st.sidebar.success(f"✅ Successfully cited segment (substitutions: {n_subs})")
            if n_subs > 0:
                cited_text = new_text
                cited_segments.add(segment_text)
                successfully_cited = True

        # Fallback: If no citations were successfully inserted, add sources at paragraph ends
        if not successfully_cited and chunks:
            cited_text = self._add_fallback_citations(cited_text, chunks)

        return cited_text
    def _insert_search_query_citations(self, text: str, gm: Optional[Dict[str, Any]]) -> str:
        """
        Alternative: Insert citations showing numbered references,
        since actual URLs might not be available in grounding metadata.
        """
        if not gm or not gm.get("grounding_supports"):
            return text
        
        supports: List[Dict[str, Any]] = gm.get("grounding_supports", [])
        
        cited_text = text
        cited_segments = set()
        citation_count = 0
        
        for s in supports:
            segment_text = s.get("text", "").strip()
            if not segment_text or segment_text in cited_segments:
                continue
            
            citation_count += 1
            # Create a simple citation marker
            citation = f' <sup style="color: #1a73e8;">[{citation_count}]</sup>'
            
            # Find and cite the segment
            seg_escaped = re.escape(segment_text)
            pattern = re.compile(rf"({seg_escaped})(?!\s*[\[\<])", flags=re.IGNORECASE)
            new_text, n_subs = pattern.subn(rf"\1{citation}", cited_text, count=1)
            
            if n_subs > 0:
                cited_text = new_text
                cited_segments.add(segment_text)
                
            # Limit to reasonable number
            if citation_count >= 10:
                break
        
                         
        return cited_text
    

    def _add_fallback_citations(self, text: str, chunks: List[Dict[str, str]]) -> str:
        """
        Fallback method: Add citation links at the end of sentences/paragraphs.
        Uses title field (actual domain) for display.
        """
        # Create unique citation links using title field
        citation_links = []
        seen_titles = set()
        
        for chunk in chunks:
            url = chunk.get("uri", "")
            title = chunk.get("title", "")
            
            if not title or not url or title in seen_titles:
                continue
            
            citation_links.append(
                f'<a href="{url}" target="_blank" '
                f'style="color: #1a73e8; text-decoration: none; font-size: 0.9em;">'
                f'{title}</a>'
            )
            seen_titles.add(title)
        
        if not citation_links:
            return text
        
        # Combine all citations
        combined_citations = " (" + ", ".join(citation_links) + ")"
        
        # Add citations after sentences ending with periods
        sentences = re.split(r'(\.["\']?\s+(?=[A-Z])|\.\s*$)', text)
        citation_count = 0
        max_citations = 3
        
        result = []
        for i, part in enumerate(sentences):
            result.append(part)
            if part.strip().endswith('.') and citation_count < max_citations:
                if i + 1 >= len(sentences) or '(' not in sentences[i + 1][:5]:
                    result.append(combined_citations)
                    citation_count += 1
        
        return ''.join(result)


    @staticmethod
    def _clean_redirects(s: str) -> str:
        """
        Remove redirect URLs, numbered citations, and normalize whitespace.
        """
        # Remove Vertex AI grounding redirect URLs
        grounding_url_pattern = r"https://vertexaisearch\.cloud\.google\.com/grounding-api-redirect/[A-Za-z0-9_-]+"
        s = re.sub(grounding_url_pattern, "", s)
        
        # Remove any numbered citations like [[1]], [1], etc.
        s = re.sub(r'\[\[?\d+\]?\]', '', s)
        
        # Normalize whitespace
        s = re.sub(r"\s+\n", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()


def main():
    st.set_page_config(
        page_title="Websearch Comparison", 
        page_icon="🔍", 
        layout="wide",
        initial_sidebar_state="expanded"  # Keep sidebar always visible
    )
    
    st.title("🔍 Websearch Comparison")
    st.caption("Adaptive, grounded answers with inline URL citations. True streaming with follow-up suggestions.")

    # SDK check
    try:
        from google import genai  # noqa: F401
        sdk_ok = True
    except ImportError:
        sdk_ok = False
        st.error("Google GenAI SDK not found. Install: pip install google-genai")
        return

    # API key check
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your-api-key-here":
        st.error("Set GEMINI_API_KEY environment variable or update GEMINI_API_KEY in this file.")
        return

    # Sidebar controls - ALWAYS VISIBLE
    st.sidebar.header("⚙️ Settings")
    
    # Model selection with corrected model names and quota information
    model_options = {
        "Flash (Recommended - Fastest)": "gemini-2.0-flash-exp",
        "Pro 2.5 (Advanced Reasoning)": "gemini-2.5-pro-preview-03-25"
        
    }
    
    selected_model_name = st.sidebar.selectbox(
        "Model",
        list(model_options.keys()),
        index=0,
        help="Flash: Fastest with high quotas (1500 RPD)\nPro 2.5: Best reasoning, higher quotas\nPro 2.0: Experimental, limited quota"
    )
    model = model_options[selected_model_name]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    
    if "Flash" in selected_model_name:
        st.sidebar.info(
            "**Flash**: Quick Analysis\n\n"
            "**Best for**: Fast responses, real-time search, general queries"
        )
    else:
        st.sidebar.info(
            "**Pro 2.5**: Thinking mode enabled\n\n"
            "**Best for**: Complex reasoning, deep analysis, coding tasks"
        )
    

    # Query input
    st.header("Query")
    query = st.text_area(
        "Enter your query",
        placeholder="e.g., Impact of Trump tariffs on India Pharma sector",
        height=120,
        key="query_input"
    )

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        go = st.button("🔍 Run", type="primary", use_container_width=True)
    with col_b:
        clear = st.button("🗑️ Clear", use_container_width=True)

    if clear:
        for k in ["last_result", "query_input"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    # Streaming display area
    stream_container = st.container()
    
    # Metrics row
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    time_badge = metric_col1.empty()
    model_badge = metric_col2.empty()
    status_badge = metric_col3.empty()

    if go and query.strip():
        # Display model being used
        if "flash" in model.lower():
            model_display = "FLASH"
        elif "2.5" in model:
            model_display = "PRO 2.5"
        else:
            model_display = "PRO 2.0"
            
        model_badge.metric("Model", model_display)
        status_badge.metric("Status", "Running")

        search = GeminiGroundingSearch(GEMINI_API_KEY)

        # True streaming with silent UI
        result = search.search_with_grounding_stream(
            query=query.strip(),
            model=model,
            container=stream_container,
            time_badge=time_badge,
            status_badge=status_badge,
        )

        st.session_state["last_result"] = result
        st.session_state["last_query"] = query.strip()

    # Display follow-up questions at the very end
    if "last_result" in st.session_state and st.session_state["last_result"].get("success"):
        result = st.session_state["last_result"]
        followup_questions = result.get("followup_questions", [])
        
        if followup_questions:
            st.markdown("---")
            st.markdown("### 💭 Suggested Follow-up Questions")
            
            cols = st.columns(len(followup_questions))
            for i, (col, question) in enumerate(zip(cols, followup_questions)):
                with col:
                    if st.button(
                        f"❓ {question}", 
                        key=f"followup_{i}",
                        use_container_width=True,
                        help="Click to ask this question"
                    ):
                        # Set the query and trigger rerun
                        st.session_state["query_input"] = question
                        st.rerun()


if __name__ == "__main__":
    main()
