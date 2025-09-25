import streamlit as st
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import os
import json
from dotenv import load_dotenv
import os
import re
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
    """Optimized Gemini search with enhanced chain prompting and dynamic header generation"""
    
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
    def _is_quality_source(uri: str, title: str) -> bool:
        """Filter for high-quality, authoritative sources"""
        quality_domains = [
            'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'cnbc.com',
            'economictimes.com', 'moneycontrol.com', 'livemint.com',
            'business-standard.com', 'financialexpress.com',
            'sebi.gov.in', 'rbi.org.in', 'cdsco.gov.in', 'fda.gov',
            'investopedia.com', 'morningstar.com', 'yahoo.com/finance',
            'google.com/finance', 'marketwatch.com', 'seeking-alpha.com'
        ]
        
        uri_lower = uri.lower()
        if any(domain in uri_lower for domain in quality_domains):
            return True
            
        quality_indicators = [
            'earnings', 'financial', 'quarterly', 'annual report', 'sec filing',
            'regulatory', 'approval', 'fda', 'sebi', 'rbi', 'cdsco',
            'stock price', 'market', 'investor', 'analysis'
        ]
        
        title_lower = title.lower()
        return any(indicator in title_lower for indicator in quality_indicators)

    @staticmethod
    
    def search_with_new_sdk(query: str) -> SearchResult:
        """Enhanced Chain-Enabled Search with Dynamic Header Generation"""
        start_time = time.time()
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            
            # Step 1: Enhanced Analysis Chain with Header Planning
            analysis_result = GeminiGroundingSearch._execute_enhanced_analysis_chain(client, query)
            
            # Step 2: Execute Content Generation Chain with Dynamic Structure
            content_result = GeminiGroundingSearch._execute_enhanced_content_chain(
                client, query, analysis_result
            )
            
            response_time = time.time() - start_time
            model_used = "gemini-2.5-flash-lite (Enhanced Chain + Dynamic Headers)"
            
            return SearchResult(
                success=True,
                response=content_result.get('response_text', ''),
                sources=content_result.get('sources', []),
                search_queries=content_result.get('search_queries', []),
                model=model_used,
                timestamp=datetime.now().isoformat(),
                response_time=response_time,
                has_grounding=content_result.get('has_grounding', False)
            )
            
        except Exception as e:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="gemini-2.5-flash (Enhanced Chain Error)",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error=str(e),
                has_grounding=False
            )


    @staticmethod
    def _execute_enhanced_analysis_chain(client, query: str) -> Dict[str, str]:
        """Enhanced analysis chain with dynamic header structure planning"""
        
        analysis_prompt = f"""Analyze this query comprehensively and design the optimal response structure: "{query}"

<analysis_framework>
1. CONTENT TYPE IDENTIFICATION:
   - current_events: Breaking news, protests, political developments requiring timeline analysis
   - business_financial: Company analysis, market data, earnings requiring metrics and performance data
   - sports_news: Match updates, scores, player performance requiring current statistics
   - technical_guide: Implementation, how-to, API documentation requiring step-by-step guidance
   - general_comprehensive: Educational, informational topics requiring thorough coverage

2. STRUCTURAL COMPLEXITY ASSESSMENT:
   - Simple: 3-4 sections, straightforward information (400-600 words)
   - Complex: 6-8 sections, multi-faceted analysis (800-1200 words)  
   - Comprehensive: 8-12 sections, extensive coverage with tables/summaries (1500-2500 words)

3. OPTIMAL HEADER STRUCTURE DETERMINATION:
   Based on content type, determine specific numbered headers that would organize information most effectively.
</analysis_framework>

<output_format>
Content_Type: [current_events|business_financial|sports_news|technical_guide|general_comprehensive]
Complexity_Level: [simple|complex|comprehensive]
Target_Length: [word count estimate]
Dynamic_Headers: Background, Timeline, Developments, Perspectives, Status, Outlook
</output_format>

Provide structural analysis:"""

        try:
            config = types.GenerateContentConfig(
                response_modalities=['TEXT'],
                max_output_tokens=3000,
                system_instruction="Analyze queries to determine optimal response structure and organization."
            )
            
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=analysis_prompt,
                config=config
            )
            
            analysis_text = response.text if response.text else "Analysis failed"
            return GeminiGroundingSearch._parse_enhanced_analysis_simple(analysis_text, query)
            
        except Exception as e:
            print(f"Enhanced analysis error: {e}")
            return GeminiGroundingSearch._create_enhanced_fallback_analysis(query)

    @staticmethod
    def _parse_enhanced_analysis_simple(analysis_text: str, query: str) -> Dict[str, str]:
        """Simplified parsing that doesn't use regex"""
        
        analysis = {
            'content_type': 'general_comprehensive',
            'complexity_level': 'complex',
            'target_length': '1000',
            'dynamic_headers': []
        }
        
        # Simple string-based parsing without regex
        lines = analysis_text.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            
            if 'content_type:' in line_lower:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    analysis['content_type'] = parts[1].strip()
            elif 'complexity_level:' in line_lower:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    analysis['complexity_level'] = parts[1].strip()
            elif 'target_length:' in line_lower:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    analysis['target_length'] = parts[1].strip()
        
        # Generate appropriate headers
        analysis['dynamic_headers'] = GeminiGroundingSearch._generate_context_aware_headers_simple(query, analysis['content_type'])
        
        return analysis

    @staticmethod
    def _generate_context_aware_headers_simple(query: str, content_type: str) -> List[str]:
        """Generate context-aware headers without regex"""
        
        query_lower = query.lower()
        
        # Simple keyword checking without regex
        current_event_keywords = ['unrest', 'protest', 'crisis', 'violence', 'election', 'conflict', 'news']
        business_keywords = ['company', 'stock', 'earnings', 'financial', 'market', 'revenue', 'business']
        sports_keywords = ['cricket', 'match', 'score', 'tournament', 'game', 'team', 'sports']
        tech_keywords = ['api', 'code', 'implementation', 'setup', 'configuration', 'technical']
        
        # Check for current events
        if content_type == 'current_events' or any(word in query_lower for word in current_event_keywords):
            return [
                "1. Background & Root Causes",
                "2. Timeline of Key Events", 
                "3. Major Developments & Escalation",
                "4. Government Response & Political Fallout",
                "5. Multiple Perspectives & Reactions",
                "6. Current Status & Recent Developments",
                "7. Regional Context & Implications",
                "8. Summary & Outlook"
            ]
        
        # Check for business/financial
        elif content_type == 'business_financial' or any(word in query_lower for word in business_keywords):
            return [
                "1. Executive Summary",
                "2. Current Financial Performance",
                "3. Market Position & Competitive Analysis", 
                "4. Recent Strategic Developments",
                "5. Key Metrics & Financial Data",
                "6. Summary table",
                "7. Growth Prospects & Challenges",
                "8. Investment Analysis & Outlook"
            ]
        
        # Check for sports
        elif content_type == 'sports_news' or any(word in query_lower for word in sports_keywords):
            return [
                "1. Current Match Status",
                "2. Key Performance Statistics",
                "3. Player & Team Analysis",
                "4. Recent Form & Head-to-Head",
                "5. Tournament Context & Standings",
                "6. Upcoming Fixtures & Predictions"
            ]
        
        # Check for technical
        elif content_type == 'technical_guide' or any(word in query_lower for word in tech_keywords):
            return [
                "1. Technical Overview",
                "2. Prerequisites & Requirements",
                "3. Step-by-Step Implementation",
                "4. Configuration & Setup",
                "5. Best Practices & Optimization",
                "6. Common Issues & Troubleshooting"
            ]
        
        # Default general structure
        else:
            return [
                "1. Overview & Background",
                "2. Key Details & Current Information",
                "3. Recent Developments & Changes",
                "4. Multiple Perspectives & Analysis",
                "5. Implications & Impact",
                "6. Future Outlook & Considerations"
            ]

    @staticmethod
    def _create_enhanced_fallback_analysis(query: str) -> Dict[str, str]:
        """Create enhanced fallback analysis when chain analysis fails"""
        
        query_lower = query.lower()
        
        # Simple keyword detection
        if any(word in query_lower for word in ['unrest', 'protest', 'crisis', 'election', 'violence', 'political']):
            content_type = 'current_events'
            target_length = '2000'
        elif any(word in query_lower for word in ['stock', 'company', 'earnings', 'financial', 'business', 'revenue']):
            content_type = 'business_financial'
            target_length = '1200'
        elif any(word in query_lower for word in ['cricket', 'match', 'score', 'sports', 'game']):
            content_type = 'sports_news'
            target_length = '600'
        elif any(word in query_lower for word in ['api', 'code', 'technical', 'implementation']):
            content_type = 'technical_guide'
            target_length = '1000'
        else:
            content_type = 'general_comprehensive'
            target_length = '1000'
        
        return {
            'content_type': content_type,
            'complexity_level': 'complex',
            'target_length': target_length,
            'dynamic_headers': GeminiGroundingSearch._generate_context_aware_headers_simple(query, content_type)
        }

    @staticmethod
    def _create_anti_duplication_prompt(query: str, analysis: Dict[str, str]) -> str:
        """Create prompt specifically designed to prevent duplication"""
        
        dynamic_headers = analysis.get('dynamic_headers', [])
        target_length = analysis.get('target_length', '1000')
        
        # Build header structure with explicit anti-duplication instructions
        headers_structure = ""
        if dynamic_headers:
            for i, header in enumerate(dynamic_headers, 1):
                # Ensure proper numbering
                if not header.startswith(f'{i}.'):
                    header = f"{i}. {header.replace('1. ', '').replace('2. ', '').replace('3. ', '').replace('4. ', '').replace('5. ', '').replace('6. ', '').replace('7. ', '').replace('8. ', '')}"
                
                headers_structure += f"""
## {header}
[Write this section ONCE and move to the next section. Include specific facts, figures, dates. Use clean citations like (source.com)]
"""
        
        # Anti-duplication prompt
        prompt = f"""CRITICAL: Use Google Search to find current 2024-2025 information for: "{query}"

ANTI-DUPLICATION RULES (MANDATORY):
- Write each numbered section header EXACTLY ONCE
- Do NOT repeat any section content anywhere in your response
- Do NOT duplicate headers like "## 1. Executive Summary" twice
- Move logically from section 1 to section 7 WITHOUT repeating
- Use clean inline citations: (reuters.com) or (cnn.com) NOT numbered lists
- Target length: ~{target_length} words TOTAL (not per section)

<response_structure>
{headers_structure}
</response_structure>

<citation_format>
- Use clean inline citations: (source.com) immediately after facts
- Do NOT use numbered citation lists like [1, 2, 3, 4, 5]
- Example: "Revenue reached $27.9 billion in FY2024 (financialexpress.com)"
</citation_format>

<strict_instructions>
- Write each section ONCE ONLY
- No duplicate headers or content
- Clean, professional formatting
- Specific data with exact figures and dates
- Move through sections 1-7 sequentially without repetition
</strict_instructions>

Generate ONE comprehensive response without ANY duplication."""

        return prompt
    @staticmethod
    def _create_clean_citation_system_instruction(analysis: Dict[str, str]) -> str:
        """Create system instruction for clean citations and no duplication"""
        
        content_type = analysis.get('content_type', 'general_comprehensive')
        
        base_instruction = """You are an expert research analyst providing comprehensive information.

CRITICAL ANTI-DUPLICATION RULES:
- Write each section header ONLY ONCE in your entire response
- Do NOT repeat section content anywhere
- Do NOT duplicate any headers like "## 1. Executive Summary"
- Move sequentially through sections without going back
- Use clean inline citations: (source.com) NOT numbered lists

CITATION FORMAT REQUIREMENTS:
- Use clean inline citations: (reuters.com), (bloomberg.com), (cnn.com)
- Do NOT use numbered citation formats like [1], [2], [3, 4, 5]
- Place citations immediately after facts: "Revenue was $10B (source.com)"
- Keep citations simple and readable

RESPONSE STRUCTURE:
- Write sections 1 through 7 sequentially
- Each section appears ONCE ONLY
- No repetition of headers or content
- Professional formatting with specific data"""

        if content_type == 'business_financial':
            return f"""{base_instruction}

BUSINESS FOCUS:
- Emphasize financial metrics and performance data
- Include exact figures, percentages, and dates
- Use financial sources: (financialexpress.com), (moneycontrol.com)
- Provide comprehensive analysis without repetition"""

        else:
            return f"""{base_instruction}

COMPREHENSIVE ANALYSIS:
- Provide balanced analysis covering all aspects
- Include current developments and implications
- Use diverse, authoritative sources with clean citations
- Structure information logically without duplication"""


    
    
    

    

    @staticmethod
    def _create_anti_duplication_prompt(query: str, analysis: Dict[str, str]) -> str:
        """Simplified prompt that prevents corruption"""
        
        dynamic_headers = analysis.get('dynamic_headers', [])
        target_length = analysis.get('target_length', '1000')
        
        # Clean header structure
        headers_structure = ""
        if dynamic_headers:
            for header in dynamic_headers:
                headers_structure += f"\n{header}\n[Provide specific information with exact figures and dates]\n"
        
        prompt = f"""Use Google Search to find current 2024-2025 information for: "{query}"

    REQUIREMENTS:
    - Write each section ONLY ONCE
    - Use exact figures with proper spacing (e.g., "$30.5 billion" not "30.5billionin")
    - Target length: {target_length} words
    - Professional business analysis format

    STRUCTURE:
    {headers_structure}

    FORMATTING:
    - Use proper number formatting: "$30.5 billion", "9.3% growth"
    - Clean section headers with ## 
    - No duplicate content
    - Specific data with sources

    Generate comprehensive analysis without duplication."""
        
        return prompt


    

    

    @staticmethod
    def _extract_sources_clean(response) -> List[Dict[str, str]]:
        """Extract sources with cleaner formatting"""
        
        sources = []
        try:
            if (response.candidates and 
                hasattr(response.candidates[0], 'grounding_metadata')):
                
                metadata = response.candidates[0].grounding_metadata
                
                if hasattr(metadata, 'grounding_chunks'):
                    seen_domains = set()
                    for chunk in metadata.grounding_chunks:
                        if (hasattr(chunk, 'web') and chunk.web and chunk.web.uri):
                            uri = chunk.web.uri
                            title = getattr(chunk.web, 'title', 'Unknown')
                            
                            # Extract clean domain name
                            try:
                                from urllib.parse import urlparse
                                domain = urlparse(uri).netloc.replace('www.', '')
                            except:
                                domain = uri.split('/')[2] if '/' in uri else uri
                            
                            # Avoid duplicate domains
                            if domain not in seen_domains:
                                # Clean title - remove extra text and truncate
                                clean_title = title.split(' - ')[0].split(' | ')[0].split('...')[0]
                                if len(clean_title) > 80:
                                    clean_title = clean_title[:80] + "..."
                                
                                sources.append({
                                    'title': clean_title,
                                    'uri': uri
                                })
                                seen_domains.add(domain)
                                
                            if len(sources) >= 8:  # Limit sources
                                break
                        
        except Exception as e:
            print(f"Source extraction error: {e}")
        
        return sources


    
    @staticmethod
    def _execute_enhanced_content_chain(client, query: str, analysis: Dict[str, str]) -> Dict[str, Any]:
        """Execute enhanced content generation with WORKING grounding"""
        
        # Simple, working prompt
        content_prompt = f"""Please provide comprehensive, current information about: "{query}"

    Use Google Search to find the most recent 2024-2025 information.

    Please organize your response with clear numbered sections:
    1. Executive Summary
    2. Current Financial Performance  
    3. Market Position & Competitive Analysis
    4. Recent Strategic Developments
    5. Key Metrics & Financial Data
    6. Summary Table if applicable
    7. Growth Prospects & Challenges
    8. Investment Analysis & Outlook

    Requirements:
    - Use real, current data from reliable sources
    - Include specific figures, dates, and statistics
    - Professional business analysis format
    - Target length: approximately 800-1000 words
    - Cite sources within your response naturally

    Please provide detailed, accurate information with proper structure."""

        system_instruction = """You are a professional research analyst providing comprehensive business intelligence.

    REQUIREMENTS:
    - Use Google Search extensively for current, factual information
    - Include specific data: exact figures, percentages, dates, company names
    - Structure responses with clear numbered sections using ## headers
    - Provide authoritative, well-researched analysis
    - Use natural source citations within your response
    - Focus on accuracy and professional presentation"""

        # Conservative token limit
        token_limit = 3000
        
        try:
            print("Debug: Starting grounding attempt...")
            
            # Use the correct grounding configuration
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            config = types.GenerateContentConfig(
                tools=[grounding_tool],
                response_modalities=['TEXT'],
                max_output_tokens=token_limit,
                system_instruction=system_instruction,
                temperature=0.1
            )
            
            print("Debug: Making API call...")
            
            # Use gemini-2.5-flash (not flash-lite) for grounding
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=content_prompt,
                config=config
            )
            
            print("Debug: Processing response...")
            
            # Simple extraction without corruption
            response_text = ""
            if hasattr(response, 'text'):
                response_text = response.text
            elif response.candidates and response.candidates[0].content.parts:
                response_text = ''.join([
                    part.text for part in response.candidates[0].content.parts 
                    if hasattr(part, 'text')
                ])
            else:
                response_text = "No response generated"
            
            # Simple source extraction
            sources = []
            search_queries = []
            has_grounding = False
            
            try:
                if (response.candidates and 
                    hasattr(response.candidates[0], 'grounding_metadata')):
                    
                    metadata = response.candidates[0].grounding_metadata
                    has_grounding = True
                    
                    if hasattr(metadata, 'grounding_chunks'):
                        for chunk in metadata.grounding_chunks:
                            if (hasattr(chunk, 'web') and chunk.web and chunk.web.uri):
                                uri = chunk.web.uri
                                title = getattr(chunk.web, 'title', 'Unknown Source')
                                
                                sources.append({
                                    'title': title[:80],  # Truncate title
                                    'uri': uri
                                })
                                
                                if len(sources) >= 8:
                                    break
                    
                    if hasattr(metadata, 'web_search_queries'):
                        search_queries = list(metadata.web_search_queries)
                        
            except Exception as e:
                print(f"Source extraction error: {e}")
            
            # MINIMAL cleanup - only fix obvious header formatting
            cleaned_response = response_text
            if response_text:
                lines = response_text.split('\n')
                cleaned_lines = []
                
                for line in lines:
                    stripped = line.strip()
                    # Only format numbered sections as headers
                    if stripped and len(stripped) < 80:  # Only short lines
                        if any(stripped.startswith(f'{i}.') for i in range(1, 9)):
                            if not stripped.startswith('##'):
                                cleaned_lines.append(f"## {stripped}")
                                continue
                    cleaned_lines.append(line)
                
                cleaned_response = '\n'.join(cleaned_lines)
                cleaned_response = cleaned_response.replace('\n## ', '\n\n## ')
            
            print(f"Debug: Success - {len(sources)} sources, {len(search_queries)} queries")
            
            return {
                'response_text': cleaned_response,
                'sources': sources,
                'search_queries': search_queries,
                'has_grounding': has_grounding,
                'analysis_used': analysis,
                'debug_info': {
                    'model_used': 'gemini-2.5-flash-lite',
                    'grounding_status': 'success' if has_grounding else 'no_metadata'
                }
            }
            
        except Exception as e:
            print(f"Grounding failed: {e}")
            
            # Simple fallback without complex processing
            try:
                print("Debug: Trying fallback...")
                
                fallback_config = types.GenerateContentConfig(
                    response_modalities=['TEXT'],
                    max_output_tokens=token_limit,
                    system_instruction=system_instruction,
                    temperature=0.2
                )
                
                fallback_response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=content_prompt,
                    config=fallback_config
                )
                
                fallback_text = ""
                if hasattr(fallback_response, 'text'):
                    fallback_text = fallback_response.text
                elif fallback_response.candidates and fallback_response.candidates[0].content.parts:
                    fallback_text = ''.join([
                        part.text for part in fallback_response.candidates[0].content.parts 
                        if hasattr(part, 'text')
                    ])
                
                return {
                    'response_text': fallback_text + "\n\n*Note: Generated without live search grounding.*",
                    'sources': [],
                    'search_queries': [query],
                    'has_grounding': False,
                    'analysis_used': analysis,
                    'debug_info': {
                        'model_used': 'gemini-2.5-flash-lite (fallback)',
                        'primary_error': str(e)
                    }
                }
                
            except Exception as fallback_error:
                print(f"Fallback failed: {fallback_error}")
                
                return {
                    'response_text': f"Error: Both primary and fallback failed. Primary: {str(e)}, Fallback: {str(fallback_error)}",
                    'sources': [],
                    'search_queries': [],
                    'has_grounding': False,
                    'analysis_used': analysis,
                    'debug_info': {
                        'primary_error': str(e),
                        'fallback_error': str(fallback_error)
                    }
                }


    @staticmethod
    def _create_simple_grounding_prompt(query: str, analysis: Dict[str, str]) -> str:
        """Create simple, clean prompt for grounding"""
        
        dynamic_headers = analysis.get('dynamic_headers', [])
        target_length = analysis.get('target_length', '1000')
        
        # Simple header list
        headers_list = ""
        if dynamic_headers:
            for header in dynamic_headers:
                headers_list += f"- {header}\n"
        
        prompt = f"""Please provide comprehensive, current information about: "{query}"

    Use Google Search to find the most recent 2024-2025 information.

    Please organize your response with these sections:
    {headers_list}

    Requirements:
    - Use real data from current sources
    - Include specific figures, dates, and statistics
    - Target length: approximately {target_length} words
    - Professional business analysis format
    - Cite sources naturally in your response

    Provide detailed, accurate information with proper structure."""

        return prompt

    @staticmethod
    def _create_simple_system_instruction(analysis: Dict[str, str]) -> str:
        """Create simple system instruction for grounding"""
        
        return """You are a professional research analyst providing comprehensive business intelligence.

    REQUIREMENTS:
    - Use Google Search extensively for current, factual information
    - Include specific data: exact figures, percentages, dates, company names
    - Structure responses with clear numbered sections
    - Provide authoritative, well-researched analysis
    - Use proper formatting with headers and bullet points
    - Cite information sources naturally within your response

    Focus on accuracy, specificity, and professional presentation."""

    @staticmethod
    def _minimal_cleanup(response_text: str) -> str:
        """Minimal cleanup to fix basic formatting without corruption"""
        
        if not response_text:
            return response_text
        
        # Only fix obvious header formatting issues
        lines = response_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Only format obvious numbered headers
            if stripped and len(stripped) < 100:  # Only short lines that could be headers
                if any(stripped.startswith(f'{i}.') for i in range(1, 10)):
                    if not stripped.startswith('##'):
                        cleaned_lines.append(f"## {stripped}")
                        continue
            
            cleaned_lines.append(line)
        
        # Join and do minimal spacing cleanup
        result = '\n'.join(cleaned_lines)
        result = result.replace('\n## ', '\n\n## ')
        
        # Remove excessive newlines only
        while '\n\n\n' in result:
            result = result.replace('\n\n\n', '\n\n')
        
        return result.strip()

        


    @staticmethod
    def _create_dynamic_nepal_style_prompt_simple(query: str, analysis: Dict[str, str]) -> str:
        """Create structured prompt without regex dependencies"""
        
        dynamic_headers = analysis.get('dynamic_headers', [])
        target_length = analysis.get('target_length', '1000')
        
        # Build header structure
        headers_structure = ""
        if dynamic_headers:
            for header in dynamic_headers:
                headers_structure += f"""
## {header}
[Provide detailed information with specific facts, figures, dates, and sources]
"""
        
        prompt = f"""IMPORTANT: Use Google Search to find current 2024-2025 information for: "{query}"

<critical_instructions>
- Write each numbered section ONLY ONCE
- Do NOT duplicate any headers or content
- Use clean, inline citations like [source.com] not numbered lists
- Target length: ~{target_length} words
- Structure as professional research report
</critical_instructions>

<requirements>
- Search for current data and recent developments
- Include specific details: dates, figures, percentages, names, locations
- Use authoritative sources and cite properly
- Target length: ~{target_length} words
- Structure as professional research report
</requirements>

<response_structure>
{headers_structure}
</response_structure>

<formatting_rules>
- Use ## for numbered headers EXACTLY ONCE (e.g., "## 1. Background & Root Causes")
- Include bullet points for detailed information
- Add specific data points with exact figures and dates
- Use clean source citations in parentheses: (source.com) or [source.com]
- Do NOT repeat sections or headers
- Maintain professional, authoritative tone throughout
</formatting_rules>

Generate comprehensive, current, well-structured content."""

        return prompt

    @staticmethod
    def _create_enhanced_system_instruction_simple(analysis: Dict[str, str]) -> str:
        """Create system instruction without complex formatting"""
        
        content_type = analysis.get('content_type', 'general_comprehensive')
        
        base_instruction = """You are an expert research analyst providing comprehensive information.

CRITICAL FORMATTING RULES:
- Write each section header ONLY ONCE
- Do NOT duplicate any content or sections
- Use clean inline citations: (reuters.com) or [cnn.com] 
- Do NOT use numbered citation lists like [1, 2, 3, 4]
- Structure responses with clear numbered sections
- Each section should flow logically without repetition

CORE REQUIREMENTS:
- Use Google Search grounding for ALL factual claims
- Include specific data: dates, figures, percentages, names, locations
- Cross-verify information from multiple sources
- Structure responses with clear numbered sections
- Maintain professional research quality"""

        if content_type == 'current_events':
            return f"""{base_instruction}

CURRENT EVENTS FOCUS:
- Provide chronological analysis with timeline
- Include multiple perspectives and reactions
- Focus on root causes and political implications
- Use credible news sources and official statements"""

        elif content_type == 'business_financial':
            return f"""{base_instruction}

BUSINESS ANALYSIS FOCUS:
- Emphasize financial metrics and performance data
- Include market analysis and competitive positioning
- Use financial sources and analyst research
- Provide investment perspective"""

        else:
            return f"""{base_instruction}

COMPREHENSIVE ANALYSIS:
- Provide balanced, multi-faceted analysis
- Include current developments and implications
- Use diverse, authoritative sources
- Structure for maximum comprehension"""

    # Simplified helper methods without regex
    @staticmethod
    def _extract_response_text_simple(response) -> str:
        """Extract response text without regex"""
        
        if hasattr(response, 'text'):
            return response.text
        elif response.candidates and response.candidates[0].content.parts:
            return ''.join([
                part.text for part in response.candidates[0].content.parts 
                if hasattr(part, 'text')
            ])
        else:
            return "No response text generated"

    @staticmethod
    def _extract_sources_simple(response) -> List[Dict[str, str]]:
        """Extract sources with cleaner formatting"""
        
        sources = []
        try:
            if (response.candidates and 
                hasattr(response.candidates[0], 'grounding_metadata')):
                
                metadata = response.candidates[0].grounding_metadata
                
                if hasattr(metadata, 'grounding_chunks'):
                    seen_urls = set()
                    for chunk in metadata.grounding_chunks:
                        if (hasattr(chunk, 'web') and chunk.web and chunk.web.uri):
                            uri = chunk.web.uri
                            title = getattr(chunk.web, 'title', 'Unknown')
                            
                            # Clean up the title and URL
                            if uri not in seen_urls:
                                # Clean title - remove extra text
                                clean_title = title.split(' - ')[0].split(' | ')[0]
                                if len(clean_title) > 60:
                                    clean_title = clean_title[:60] + "..."
                                
                                sources.append({
                                    'title': clean_title,
                                    'uri': uri
                                })
                                seen_urls.add(uri)
                                
                            if len(sources) >= 10:  # Limit sources
                                break
                        
        except Exception as e:
            print(f"Source extraction error: {e}")
        
        return sources

    

    @staticmethod
    def _extract_search_queries_simple(response) -> List[str]:
        """Extract search queries simply"""
        
        search_queries = []
        try:
            if (response.candidates and 
                hasattr(response.candidates[0], 'grounding_metadata')):
                
                metadata = response.candidates[0].grounding_metadata
                
                if hasattr(metadata, 'web_search_queries'):
                    search_queries = list(metadata.web_search_queries)
                        
        except Exception as e:
            print(f"Search query extraction error: {e}")
        
        return search_queries

    @staticmethod
    def _post_process_simple(response_text: str, analysis: Dict[str, str]) -> str:
        """Simple post-processing without regex - fixed to prevent duplication"""
        
        if not response_text:
            return response_text
        
        # Split into lines and remove any existing duplicates
        lines = response_text.split('\n')
        processed_lines = []
        seen_headers = set()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines at the beginning
            if not line and not processed_lines:
                i += 1
                continue
            
            # Check for numbered headers
            if line and any(line.startswith(f'{j}.') for j in range(1, 10)):
                # Check if this header was already seen
                header_key = line.split('.')[0] + '.'
                if header_key in seen_headers:
                    # Skip this duplicate section entirely
                    i += 1
                    # Skip until next header or end
                    while i < len(lines) and not any(lines[i].strip().startswith(f'{k}.') for k in range(1, 10)):
                        i += 1
                    continue
                
                seen_headers.add(header_key)
                
                # Format header properly
                if not line.startswith('##'):
                    processed_lines.append(f"## {line}")
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)
            
            i += 1
        
        # Reconstruct and clean up
        formatted_response = '\n'.join(processed_lines)
        
        # Fix spacing
        formatted_response = formatted_response.replace('\n## ', '\n\n## ')
        
        # Remove excessive newlines
        while '\n\n\n' in formatted_response:
            formatted_response = formatted_response.replace('\n\n\n', '\n\n')
        
        return formatted_response.strip()


   

    

    

    

      
    
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
                
                # Use only Gemini 2.5 Flash-lite
                model = genai_old.GenerativeModel("gemini-2.5-flash-lite")
                
                # Try grounding first, fallback to basic
                try:
                    if TOOL_CONFIG_AVAILABLE:
                        response = model.generate_content(
                            prompt,
                            tools=[{'google_search_retrieval': {}}]
                        )
                        model_used = "gemini-2.5-flash-lite (Legacy Grounding)"
                    else:
                        response = model.generate_content(prompt)
                        model_used = "gemini-2.5-flash-lite (Legacy Basic)"
                except Exception:
                    # Fallback to basic
                    response = model.generate_content(prompt)
                    model_used = "gemini-2.5-flash-lite (Legacy Basic)"

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
                    model="gemini-2.5-flash-lite (Legacy Error)",
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
    def enhance_query_for_azure(query: str) -> str:
        """Create comprehensive, structured prompt for Azure AI Agents"""
        return f"""
        You are a professional research analyst and business intelligence expert. Provide a comprehensive, 
        detailed analysis for: "{query}"
        
        RESEARCH REQUIREMENTS:
        - Search for the most current information available (prioritize 2024-2025 data)
        - Include specific financial figures, percentages, dates, and quantitative metrics
        - Provide company-by-company or topic-by-topic detailed breakdown
        - Include recent developments, regulatory changes, and market trends
        - Add strategic insights and forward-looking analysis
        - Compare multiple perspectives and sources when relevant
        - User location: India (prioritize Indian market context)
        
        RESPONSE STRUCTURE REQUIRED:
        1. **Executive Summary** (2-3 sentences with key findings)
        2. **Detailed Analysis** with subheadings for each major point
        3. **Key Metrics & Data** (use specific numbers, percentages, dates)
        4. **Recent Developments** (last 6-12 months)
        5. **Strategic Implications** and outlook
        6. **Summary Table** ( wherever applicable )
        7. **Actionable Recommendation** ( wherever applicable )
        8. **Sources & Citations** (properly reference all claims)
        
        SEARCH INSTRUCTION: Use Bing grounding tool to find the most current, authoritative sources 
        including financial reports, regulatory filings, industry analyses, and recent news.
        """

    
    def create_enhanced_instructions(query: str) -> str:
        """Create detailed additional instructions for the Azure AI Agent"""
        return f"""
        CRITICAL SEARCH INSTRUCTIONS for query: "{query}"

        **MANDATORY ACTIONS:**
        1. MUST use Bing search grounding tool for current information
        2. MUST search for 2024-2025 specific data and developments  
        3. MUST include quantitative metrics (percentages, figures, dates)
        4. MUST provide company-specific or sector-specific breakdown
        5. MUST cite authoritative sources for all claims

        **CONTENT DEPTH REQUIREMENTS:**
        - Minimum 500-800 words for comprehensive coverage
        - Include specific financial data, market metrics, regulatory updates
        - Provide both current performance data AND forward-looking insights
        - Compare multiple companies/aspects when relevant
        
        **BUSINESS INTELLIGENCE FORMAT:**
        Structure the response as a professional business intelligence report suitable for:
        - Investment decision-making
        - Strategic business planning  
        - Regulatory compliance assessment
        - Market analysis and competitive intelligence

        **ADDITIONAL REQUIREMENTS:**
        - Include visual aids (charts, graphs) where applicable
        - Ensure clarity and conciseness in language
        - Tailor content to the target audience's expertise level

        This query requires REAL-TIME web search data. Do not rely on training data alone.
        """

    
    def search(query: str) -> SearchResult:
        """Search using Azure AI Foundry Agent Service with Bing grounding via REST API"""
        start_time = time.time()
        
        # Import required libraries
        try:
            import streamlit as st
            import requests
            from azure.identity import ClientSecretCredential, DefaultAzureCredential
            from azure.core.exceptions import ClientAuthenticationError
            import os
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
        
        # Get configuration from Streamlit secrets
        try:
            # Extract project endpoint and project name from the full endpoint
            full_endpoint = st.secrets["AZURE_AI_FOUNDRY_ENDPOINT"]
            agent_id = st.secrets.get("AZURE_AGENT_ID", "")
            tenant_id = st.secrets.get("AZURE_TENANT_ID", "")
            
            # For service principal authentication (required for Streamlit Cloud)
            client_id = st.secrets.get("AZURE_CLIENT_ID", "")
            client_secret = st.secrets.get("AZURE_CLIENT_SECRET", "")
            
            # Extract the correct endpoint format
            if "/api/projects/" in full_endpoint:
                base_endpoint = full_endpoint.split("/api/projects/")[0]
                project_path_part = full_endpoint.split("/api/projects/")[1].rstrip('/')
                project_endpoint = f"{base_endpoint}/api/projects/{project_path_part}"
            else:
                base_endpoint = full_endpoint.rstrip('/')
                if ".services.ai.azure.com" in base_endpoint:
                    subdomain = base_endpoint.split("//")[1].split(".")[0]
                    project_name = f"{subdomain}_project"
                    project_endpoint = f"{base_endpoint}/api/projects/{project_name}"
                else:
                    project_endpoint = base_endpoint
            
        except KeyError as e:
            return SearchResult(
                success=False,
                response="",
                sources=[],
                search_queries=[],
                model="Azure Configuration Missing",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error=f"Missing Streamlit secret: {e}. Need AZURE_AI_FOUNDRY_ENDPOINT, AZURE_AGENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET",
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
                search_queries=[],
                model="Azure Agent Not Configured",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error="AZURE_AGENT_ID not configured. Please create an agent in Azure AI Foundry first.",
                has_grounding=False
            )
        
        try:
            # Authentication with CORRECT scope for Azure AI Foundry
            credential = None
            
            if client_id and client_secret and tenant_id:
                # Method 1: Service Principal Authentication (Recommended for Streamlit Cloud)
                try:
                    credential = ClientSecretCredential(
                        tenant_id=tenant_id,
                        client_id=client_id,
                        client_secret=client_secret
                    )
                    
                    # Try multiple scopes to find the correct one
                    token_scopes = [
                        "https://ai.azure.com/.default",  # Primary scope for AI Foundry
                        "https://cognitiveservices.azure.com/.default",  # Fallback scope
                        "https://management.azure.com/.default"  # Management scope
                    ]
                    
                    token_result = None
                    successful_scope = None
                    
                    for scope in token_scopes:
                        try:
                            token_result = credential.get_token(scope)
                            successful_scope = scope
                            break
                        except Exception:
                            continue
                    
                    if not token_result:
                        return SearchResult(
                            success=False,
                            response="",
                            sources=[],
                            search_queries=[],
                            model="Token Acquisition Failed",
                            timestamp=datetime.now().isoformat(),
                            response_time=time.time() - start_time,
                            error="Failed to acquire token with any of the attempted scopes. Please verify service principal permissions.",
                            has_grounding=False
                        )
                    
                    headers = {
                        "Authorization": f"Bearer {token_result.token}",
                        "Content-Type": "application/json",
                        "User-Agent": "StreamlitApp/1.0"
                    }
                    
                except ClientAuthenticationError as auth_error:
                    return SearchResult(
                        success=False,
                        response="",
                        sources=[],
                        search_queries=[],
                        model="Service Principal Auth Failed",
                        timestamp=datetime.now().isoformat(),
                        response_time=time.time() - start_time,
                        error=f"Service Principal authentication failed: {auth_error}. Please verify your AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, and AZURE_TENANT_ID.",
                        has_grounding=False
                    )
                except Exception as auth_error:
                    return SearchResult(
                        success=False,
                        response="",
                        sources=[],
                        search_queries=[],
                        model="Authentication Setup Failed",
                        timestamp=datetime.now().isoformat(),
                        response_time=time.time() - start_time,
                        error=f"Authentication setup failed: {auth_error}",
                        has_grounding=False
                    )
            
            else:
                # Method 2: DefaultAzureCredential (fallback for local development)
                try:
                    # Set environment variables for DefaultAzureCredential if available
                    if tenant_id:
                        os.environ['AZURE_TENANT_ID'] = tenant_id
                    if client_id:
                        os.environ['AZURE_CLIENT_ID'] = client_id
                    if client_secret:
                        os.environ['AZURE_CLIENT_SECRET'] = client_secret
                    
                    credential = DefaultAzureCredential()
                    
                    # Try the same multiple scopes approach
                    token_scopes = [
                        "https://ai.azure.com/.default",
                        "https://cognitiveservices.azure.com/.default",
                        "https://management.azure.com/.default"
                    ]
                    
                    token_result = None
                    successful_scope = None
                    
                    for scope in token_scopes:
                        try:
                            token_result = credential.get_token(scope)
                            successful_scope = scope
                            break
                        except Exception:
                            continue
                    
                    if not token_result:
                        return SearchResult(
                            success=False,
                            response="",
                            sources=[],
                            search_queries=[],
                            model="Default Auth Failed",
                            timestamp=datetime.now().isoformat(),
                            response_time=time.time() - start_time,
                            error="Authentication failed. For Streamlit Cloud deployment, you need to add AZURE_CLIENT_ID and AZURE_CLIENT_SECRET to your secrets.",
                            has_grounding=False
                        )
                    
                    headers = {
                        "Authorization": f"Bearer {token_result.token}",
                        "Content-Type": "application/json",
                        "User-Agent": "StreamlitApp/1.0"
                    }
                    
                except ClientAuthenticationError as auth_error:
                    return SearchResult(
                        success=False,
                        response="",
                        sources=[],
                        search_queries=[],
                        model="Default Auth Failed",
                        timestamp=datetime.now().isoformat(),
                        response_time=time.time() - start_time,
                        error=f"Authentication failed. For Streamlit Cloud deployment, you need to add AZURE_CLIENT_ID and AZURE_CLIENT_SECRET to your secrets. Error: {auth_error}",
                        has_grounding=False
                    )
            
            # Enhanced query for better results
            enhanced_query = AzureAIAgentsSearch.enhance_query_for_azure(query)

            enhanced_instructions = AzureAIAgentsSearch.create_enhanced_instructions(query)
            
            # Use the correct API version for Azure AI Foundry Agent Service
            api_version = "2025-05-01"  # Stable version for Agent Service
            
            # Step 1: Create thread with correct endpoint format
            thread_url = f"{project_endpoint}/threads"
            thread_params = {"api-version": api_version}
            
            thread_response = requests.post(
                thread_url, 
                headers=headers, 
                json={}, 
                params=thread_params, 
                timeout=30
            )
            
            if thread_response.status_code != 200:
                error_details = thread_response.text
                
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
                        error=f"Authentication failed (401). Token audience mismatch. Please ensure your Service Principal has proper permissions for Azure AI Foundry. Details: {error_details}",
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
                        error=f"Access forbidden (403). Please ensure your Service Principal has these Azure roles: Cognitive Services User, Azure AI Developer. Details: {error_details}",
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
                "additional_instructions": enhanced_instructions
            }

            run_response = requests.post(
                run_url,
                headers=headers,
                json=run_data,
                params=thread_params,
                timeout=30
            )

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

            # Step 4: Poll for completion
            run_status_url = f"{project_endpoint}/threads/{thread_id}/runs/{run_id}"
            max_polls = 60  # 2 minutes max
            poll_count = 0

            while poll_count < max_polls:
                try:
                    status_response = requests.get(
                        run_status_url,
                        headers=headers,
                        params=thread_params,
                        timeout=15
                    )
                    
                    if status_response.status_code != 200:
                        break
                    
                    status_data = status_response.json()
                    status = status_data.get("status")
                    
                    if status == "completed":
                        break
                    elif status in ["failed", "cancelled", "expired"]:
                        error_details = status_data.get("last_error", {})
                        error_msg = error_details.get("message", "Unknown error")
                        return SearchResult(
                            success=False,
                            response="",
                            sources=[],
                            search_queries=[],
                            model=f"Azure AI Agent ({agent_id})",
                            timestamp=datetime.now().isoformat(),
                            response_time=time.time() - start_time,
                            error=f"Run failed with status '{status}': {error_msg}",
                            has_grounding=False
                        )
                    
                    time.sleep(2)
                    poll_count += 1
                    
                except requests.RequestException:
                    time.sleep(3)
                    poll_count += 1

            if poll_count >= max_polls:
                return SearchResult(
                    success=False,
                    response="",
                    sources=[],
                    search_queries=[],
                    model=f"Azure AI Agent ({agent_id})",
                    timestamp=datetime.now().isoformat(),
                    response_time=time.time() - start_time,
                    error="Run timed out waiting for completion after 2 minutes",
                    has_grounding=False
                )

            # Step 5: Get messages from thread
            messages_url = f"{project_endpoint}/threads/{thread_id}/messages"

            messages_response = requests.get(
                messages_url,
                headers=headers,
                params=dict(thread_params, order="asc"),
                timeout=30
            )

            if messages_response.status_code != 200:
                return SearchResult(
                    success=False,
                    response="",
                    sources=[],
                    search_queries=[],
                    model=f"Azure AI Agent ({agent_id})",
                    timestamp=datetime.now().isoformat(),
                    response_time=time.time() - start_time,
                    error=f"Failed to get messages: {messages_response.status_code} - {messages_response.text}",
                    has_grounding=False
                )

            messages_data = messages_response.json()

            # Step 6: Parse response
            response_text = ""
            sources = []
            search_queries = [query]
            has_grounding = False

            # Extract the assistant's response (most recent assistant message)
            messages_list = messages_data.get("data", [])
            assistant_messages = [msg for msg in messages_list if msg.get("role") == "assistant"]

            if assistant_messages:
                latest_message = assistant_messages[-1]
                content_items = latest_message.get("content", [])
                
                for content_item in content_items:
                    if content_item.get("type") == "text":
                        text_data = content_item.get("text", {})
                        response_text = text_data.get("value", "")
                        
                        # Extract citations/sources from annotations
                        annotations = text_data.get("annotations", [])
                        
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
                                # Handle URL citations from Bing search
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

            # Heuristic grounding detection
            if not has_grounding and response_text:
                grounding_indicators = [
                    "according to", "based on recent", "current information",
                    "latest", "recent reports", "today", "this year", "2024", "2025",
                    "search results show", "found that", "indicates", "sources suggest",
                    "according to web sources", "recent data", "current data", "bing search"
                ]
                
                response_lower = response_text.lower()
                if any(indicator in response_lower for indicator in grounding_indicators):
                    has_grounding = True

            # Clean up response
            if not response_text:
                response_text = "No response generated by the agent."

            response_time = time.time() - start_time

            return SearchResult(
                success=True,
                response=response_text,
                sources=sources[:10],  # Limit to 10 sources
                search_queries=search_queries,
                model=f"Azure AI Foundry Agent ({agent_id}) with Bing Grounding",
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
                model="Azure AI Error",
                timestamp=datetime.now().isoformat(),
                response_time=time.time() - start_time,
                error=str(e),
                has_grounding=False
            )

def add_citations_to_text(response_result: SearchResult) -> str:
    """Add inline citations to the response text for Gemini and Azure AI Agents"""
    if not response_result.has_grounding or not response_result.sources:
        return response_result.response
    
    # Apply inline citations to Gemini models and Azure AI Agents
    model_lower = response_result.model.lower()
    if "gemini" not in model_lower and "azure" not in model_lower:
        return response_result.response
    
    text = response_result.response
    sources = response_result.sources
    
    if not sources:
        return text
    
    # For Azure AI Agents, handle the special citation format
    if "azure" in model_lower:
        import re
        
        # Replace Azure citation patterns with inline links
        # Pattern matches 【3:1†source】, 【3:2†source】, etc.
        citation_pattern = r'【\d+:\d+†source】'
        
        # Find all citations in the text
        citations = re.findall(citation_pattern, text)
        
        # Replace each citation with corresponding source link
        source_index = 0
        for citation in citations:
            if source_index < len(sources):
                source = sources[source_index]
                title = source.get('title', f'Source {source_index + 1}')
                uri = source.get('uri', '')
                
                if uri:
                    replacement = f" [{title}]({uri})"
                    text = text.replace(citation, replacement, 1)  # Replace only first occurrence
                else:
                    text = text.replace(citation, f" [{title}]", 1)
                
                source_index += 1
            else:
                # Remove citation if no corresponding source
                text = text.replace(citation, '', 1)
        
        return text
    
    # For Gemini models, use the original paragraph-based approach
    else:
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
        options.append("Gemini 2.5 Flash-lite with Google Search Grounding")
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