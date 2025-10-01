# file: GEMINI_GPT_V1.py

import os
import re
import time
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
import streamlit as st

# Configure API key: recommended to set GEMINI_API_KEY in the environment
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]  # replace if needed

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
        Adaptive prompt that only includes relevant sections per query type.
        """
        return f"""
Analyze the following query and provide a comprehensive response using up-to-date grounded information.

Query: {query}

ADAPTIVE STRUCTURE:
- ALWAYS: ## Executive Summary (2-3 sentences), ## Main Analysis (well-structured core content)
-Use ##Executive summary or ##Overview interchangeably depending on the query type. If formal query, business query use "Executive Summary" but in case of informal query or more personal interest query start with "Overview".
- INCLUDE WHEN RELEVANT: Key Facts & Figures, Timeline, Comparative Analysis (tables), Regional/Geographic Breakdown,
  Current Status, Technical Details, Impact Analysis, Future Outlook, Recommendations
- EXCLUDE IRRELEVANT: Do not include sections that do not add value for this query type
  (e.g., recommendations for sports results; future outlook for concluded historical events).

QUERY TYPE HINTS:
- Factual/Informational → concise facts, dates, numbers
- Sports/Entertainment → scores, highlights, player performance (no recommendations/outlook)
- Comparison/Market → tables, metrics, analysis; recommendations if decision-oriented
- How-to/Technical → step-by-step, technical details
- Current Events → timeline + current status + recent developments
- Historical → timeline + impact (no predictions)

FORMATTING:
- Use ## for main sections, ### for subsections
- Use **bold** for key terms and numbers
- Use markdown tables for comparisons
- Use bullet points (- or *) for lists
- DO NOT include any citation numbers like [1], [2] in your response - citations will be added automatically
- Ensure the response is complete and not truncated
""".strip()

    def _shorten_url_domain(self, url: str) -> str:
        """
        Extract and shorten URL to display format like 'example.com'
        """
        try:
            # Skip vertex AI redirect URLs
            if "vertexaisearch.cloud.google.com" in url:
                return None
                
            parsed = urlparse(url)
            netloc = parsed.netloc
            
            # Remove 'www.' prefix if present
            if netloc.startswith('www.'):
                netloc = netloc[4:]
            
            # Keep just domain.tld for cleaner display
            parts = netloc.split('.')
            if len(parts) >= 2:
                netloc = '.'.join(parts[-2:])
            
            return netloc if netloc else None
        except:
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
        temperature: float = 0.7,
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

            # Insert inline citations with clickable shortened URL links immediately after streaming
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
                            if hasattr(ch, "web") and ch.web:
                                chunks.append({
                                    "uri": getattr(ch.web, "uri", ""),
                                    "title": getattr(ch.web, "title", ""),
                                })
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

                    return meta
            return None
        except Exception:
            return None

    def _insert_inline_url_citations(self, text: str, gm: Optional[Dict[str, Any]]) -> str:
        """
        Insert inline citations as clickable shortened URL links (e.g., 'example.com')
        - Shows shortened domain names as clickable links inline
        - Multiple sources shown as comma-separated links
        """
        if not gm or not gm.get("grounding_chunks"):
            return self._clean_redirects(text)

        chunks: List[Dict[str, str]] = gm.get("grounding_chunks", [])
        supports: List[Dict[str, Any]] = gm.get("grounding_supports", [])

        cited_text = self._clean_redirects(text)
        cited_segments = set()

        # For each supported segment, append shortened URL citations
        for s in supports:
            segment_text = s.get("text", "").strip()
            if not segment_text or segment_text in cited_segments:
                continue

            chunk_indices = s.get("grounding_chunk_indices", [])
            if not chunk_indices:
                continue

            # Create citation with shortened URLs
            citation_parts = []
            for idx in chunk_indices:
                if idx < len(chunks):
                    url = chunks[idx].get("uri", "")
                    title = chunks[idx].get("title", "Source")
                    
                    shortened_display = self._shorten_url_domain(url)
                    if not shortened_display:
                        continue
                    
                    # Create HTML link with shortened domain as display text
                    citation_parts.append(
                        f'<a href="{url}" target="_blank" title="{title}" '
                        f'style="text-decoration:none;color:#1a73e8;font-size:0.85em;font-weight:500;">'
                        f'{shortened_display}</a>'
                    )
            
            if not citation_parts:
                continue
            
            # Join multiple sources with commas in parentheses
            citation = " <span style='color:#666;font-size:0.85em;'>(" + ", ".join(citation_parts) + ")</span>"

            # Find first occurrence of segment not already cited
            seg_escaped = re.escape(segment_text)
            pattern = re.compile(rf"({seg_escaped})(?!\s*[\(\<])", flags=re.IGNORECASE)
            
            new_text, n_subs = pattern.subn(rf"\1{citation}", cited_text, count=1)
            
            if n_subs > 0:
                cited_text = new_text
                cited_segments.add(segment_text)

        return cited_text

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
        "Pro 2.5 (Advanced Reasoning)": "gemini-2.5-pro-preview-03-25",
        "Pro 2.0 Exp (Experimental)": "gemini-2.0-pro-exp"
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
            "**Flash**: 192+ tokens/sec\n\n"
            "**Quota**: 1,500 requests/day (free)\n\n"
            "**Best for**: Fast responses, real-time search, general queries"
        )
    elif "2.5" in selected_model_name:
        st.sidebar.info(
            "**Pro 2.5**: Thinking mode enabled\n\n"
            "**Quota**: Higher limits than 2.0\n\n"
            "**Best for**: Complex reasoning, deep analysis, coding tasks"
        )
    else:
        st.sidebar.warning(
            "**Pro 2.0 Exp**: Experimental\n\n"
            "**Quota**: Limited (may hit rate limits)\n\n"
            "**Note**: Switch to Pro 2.5 or Flash if you see quota errors"
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
