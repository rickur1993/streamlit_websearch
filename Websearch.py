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
    def _post_process_response(response_text: str) -> str:
        """Enhance response structure if agent didn't follow instructions properly"""
        if not response_text or "## Executive Summary" in response_text:
            return response_text
            
        lines = response_text.split('\n')
        if len(lines) > 2 and not response_text.startswith('##'):
            structured_response = f"## Executive Summary\n\n{lines[0]}\n\n## Detailed Analysis\n\n"
            structured_response += '\n'.join(lines[1:])
            return structured_response
            
        return response_text

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
                model="gemini-2.5-flash-lite (Enhanced Chain Error)",
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
   Based on content type, determine specific numbered headers that would organize information most effectively:
   
   For current_events: Background & Causes → Timeline → Key Developments → Multiple Perspectives → Current Status → Outlook
   For business_financial: Executive Summary → Financial Performance → Market Analysis → Recent Developments → Strategic Outlook
   For sports_news: Current Status → Match Details → Performance Analysis → Recent Form → Looking Ahead
   For technical_guide: Overview → Implementation → Configuration → Best Practices → Troubleshooting
   For general_comprehensive: Overview → Key Information → Current Developments → Analysis → Implications
</analysis_framework>

<output_format>
Content_Type: [current_events|business_financial|sports_news|technical_guide|general_comprehensive]
Complexity_Level: [simple|complex|comprehensive]
Header_Count: [number between 3-12]
Dynamic_Headers: [List specific numbered headers like "1. Background & Root Causes", "2. Timeline of Events", etc.]
Special_Elements: [summary_table|timeline|comparison_analysis|multiple_perspectives|data_metrics]
Target_Length: [word count estimate]
Key_Focus_Areas: [list 3-4 main aspects to emphasize]
</output_format>

Provide detailed structural blueprint:"""

        try:
            config = types.GenerateContentConfig(
                response_modalities=['TEXT'],
                max_output_tokens=500,
                system_instruction="""You are an expert content strategist specializing in dynamic content organization. 
                Analyze queries to determine optimal response structure, specific header sequences, and information hierarchy. 
                Focus on creating numbered section headers that logically organize complex information like professional research reports."""
            )
            
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=analysis_prompt,
                config=config
            )
            
            analysis_text = response.text if response.text else "Analysis failed"
            return GeminiGroundingSearch._parse_enhanced_analysis(analysis_text, query)
            
        except Exception as e:
            print(f"Enhanced analysis error: {e}")
            return GeminiGroundingSearch._create_enhanced_fallback_analysis(query)

    @staticmethod
    def _parse_enhanced_analysis(analysis_text: str, query: str) -> Dict[str, str]:
        """Parse enhanced analysis response with dynamic header extraction"""
        
        analysis = {
            'content_type': 'general_comprehensive',
            'complexity_level': 'complex',
            'header_count': '6',
            'dynamic_headers': [],
            'special_elements': [],
            'target_length': '1000',
            'key_focus_areas': []
        }
        
        lines = analysis_text.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            
            if 'content_type:' in line_lower:
                analysis['content_type'] = line.split(':', 1)[1].strip()
            elif 'complexity_level:' in line_lower:
                analysis['complexity_level'] = line.split(':', 1)[1].strip()
            elif 'header_count:' in line_lower:
                analysis['header_count'] = line.split(':', 1)[1].strip()
            elif 'dynamic_headers:' in line_lower:
                headers_section = line.split(':', 1)[1].strip()
                # Extract headers from bracketed list or line by line
                if '[' in headers_section and ']' in headers_section:
                    headers_text = headers_section.strip('[]')
                    analysis['dynamic_headers'] = [h.strip().strip('"') for h in headers_text.split(',')]
                else:
                    # Look for headers in subsequent lines
                    header_lines = []
                    line_idx = lines.index(line)
                    for subsequent_line in lines[line_idx+1:line_idx+15]:  # Look ahead for headers
                        if re.match(r'^\d+\.', subsequent_line.strip()):
                            header_lines.append(subsequent_line.strip())
                    analysis['dynamic_headers'] = header_lines
            elif 'special_elements:' in line_lower:
                elements_text = line.split(':', 1)[1].strip()
                if '[' in elements_text and ']' in elements_text:
                    elements_text = elements_text.strip('[]')
                    analysis['special_elements'] = [e.strip() for e in elements_text.split(',')]
            elif 'target_length:' in line_lower:
                analysis['target_length'] = line.split(':', 1)[1].strip()
            elif 'key_focus_areas:' in line_lower:
                focus_text = line.split(':', 1)[1].strip()
                if '[' in focus_text and ']' in focus_text:
                    focus_text = focus_text.strip('[]')
                    analysis['key_focus_areas'] = [f.strip() for f in focus_text.split(',')]
        
        # Generate fallback headers if parsing failed
        if not analysis['dynamic_headers']:
            analysis['dynamic_headers'] = GeminiGroundingSearch._generate_context_aware_headers(query, analysis['content_type'])
        
        return analysis

    @staticmethod
    def _generate_context_aware_headers(query: str, content_type: str) -> List[str]:
        """Generate context-aware headers based on query content and type"""
        
        query_lower = query.lower()
        
        # Current events pattern (like Nepal Gen Z example)
        if content_type == 'current_events' or any(word in query_lower for word in ['unrest', 'protest', 'crisis', 'violence', 'election', 'conflict']):
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
        
        # Business/Financial analysis
        elif content_type == 'business_financial' or any(word in query_lower for word in ['company', 'stock', 'earnings', 'financial', 'market', 'revenue']):
            return [
                "1. Executive Summary",
                "2. Current Financial Performance",
                "3. Market Position & Competitive Analysis", 
                "4. Recent Strategic Developments",
                "5. Key Metrics & Financial Data",
                "6. Growth Prospects & Challenges",
                "7. Investment Analysis & Outlook"
            ]
        
        # Sports/News updates
        elif content_type == 'sports_news' or any(word in query_lower for word in ['cricket', 'match', 'score', 'tournament', 'game', 'team']):
            return [
                "1. Current Match Status",
                "2. Key Performance Statistics",
                "3. Player & Team Analysis",
                "4. Recent Form & Head-to-Head",
                "5. Tournament Context & Standings",
                "6. Upcoming Fixtures & Predictions"
            ]
        
        # Technical guides
        elif content_type == 'technical_guide' or any(word in query_lower for word in ['api', 'code', 'implementation', 'setup', 'configuration']):
            return [
                "1. Technical Overview",
                "2. Prerequisites & Requirements",
                "3. Step-by-Step Implementation",
                "4. Configuration & Setup",
                "5. Best Practices & Optimization",
                "6. Common Issues & Troubleshooting",
                "7. Advanced Features & Extensions"
            ]
        
        # General comprehensive information
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
        
        # Detect content type through keywords
        if any(word in query_lower for word in ['unrest', 'protest', 'crisis', 'election', 'violence', 'political']):
            content_type = 'current_events'
            complexity = 'comprehensive'
            target_length = '2000'
        elif any(word in query_lower for word in ['stock', 'company', 'earnings', 'financial', 'business', 'revenue']):
            content_type = 'business_financial'
            complexity = 'complex'
            target_length = '1200'
        elif any(word in query_lower for word in ['cricket', 'match', 'score', 'sports', 'game']):
            content_type = 'sports_news'
            complexity = 'simple'
            target_length = '600'
        elif any(word in query_lower for word in ['api', 'code', 'technical', 'implementation']):
            content_type = 'technical_guide'
            complexity = 'complex'
            target_length = '1000'
        else:
            content_type = 'general_comprehensive'
            complexity = 'complex'
            target_length = '1000'
        
        return {
            'content_type': content_type,
            'complexity_level': complexity,
            'header_count': '6',
            'dynamic_headers': GeminiGroundingSearch._generate_context_aware_headers(query, content_type),
            'special_elements': ['summary_table'] if content_type in ['current_events', 'business_financial'] else [],
            'target_length': target_length,
            'key_focus_areas': ['current information', 'detailed analysis', 'multiple perspectives']
        }

    @staticmethod
    def _execute_enhanced_content_chain(client, query: str, analysis: Dict[str, str]) -> Dict[str, Any]:
        """Execute enhanced content generation with fixed grounding and dynamic structure"""
        
        # Generate dynamic structured prompt
        content_prompt = GeminiGroundingSearch._create_dynamic_nepal_style_prompt(query, analysis)
        system_instruction = GeminiGroundingSearch._create_enhanced_system_instruction(analysis)
        
        # Calculate adaptive token limit
        target_length = analysis.get('target_length', '1000')
        token_limit = GeminiGroundingSearch._get_adaptive_token_limit(analysis)
        
        # Setup grounding tool with correct configuration for Gemini 2.5
        try:
            # Use the correct grounding tool configuration
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            config = types.GenerateContentConfig(
                tools=[grounding_tool],
                response_modalities=['TEXT'],
                max_output_tokens=token_limit,
                system_instruction=system_instruction,
                temperature=0.05  # Very low temperature for consistent structured output
            )
            
            print(f"Debug: Using token limit: {token_limit}")
            print(f"Debug: Target length: {target_length}")
            print(f"Debug: Content type: {analysis.get('content_type', 'unknown')}")
            
            # Execute content generation
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=content_prompt,
                config=config
            )
            
            # Debug grounding status
            has_grounding_metadata = False
            if response.candidates and hasattr(response.candidates[0], 'grounding_metadata'):
                has_grounding_metadata = True
                print("Debug: Grounding metadata found")
            else:
                print("Debug: No grounding metadata - checking response structure")
            
            # Extract and process response
            response_text = GeminiGroundingSearch._extract_response_text(response)
            sources = GeminiGroundingSearch._extract_sources_with_quality_filter(response)
            search_queries = GeminiGroundingSearch._extract_search_queries(response)
            has_grounding = len(sources) > 0
            
            print(f"Debug: Extracted {len(sources)} sources, {len(search_queries)} search queries")
            
            # Post-process for Nepal-style structure
            processed_response = GeminiGroundingSearch._post_process_nepal_style_response(
                response_text, analysis
            )
            
            return {
                'response_text': processed_response,
                'sources': sources,
                'search_queries': search_queries,
                'has_grounding': has_grounding,
                'analysis_used': analysis,
                'structure_type': 'dynamic_nepal_style',
                'debug_info': {
                    'token_limit': token_limit,
                    'has_metadata': has_grounding_metadata,
                    'response_length': len(response_text),
                    'sources_count': len(sources)
                }
            }
            
        except Exception as e:
            print(f"Enhanced content chain error: {e}")
            print(f"Error type: {type(e)}")
            
            # Fallback without grounding if grounding fails
            try:
                print("Debug: Attempting fallback without grounding...")
                
                fallback_config = types.GenerateContentConfig(
                    response_modalities=['TEXT'],
                    max_output_tokens=token_limit,
                    system_instruction=system_instruction + "\n\nNote: Operating without Google Search grounding. Provide the best possible response based on training data.",
                    temperature=0.1
                )
                
                fallback_response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=content_prompt,
                    config=fallback_config
                )
                
                fallback_text = GeminiGroundingSearch._extract_response_text(fallback_response)
                processed_fallback = GeminiGroundingSearch._post_process_nepal_style_response(
                    fallback_text, analysis
                )
                
                return {
                    'response_text': processed_fallback + "\n\n*Note: This response was generated without live search grounding due to technical limitations.*",
                    'sources': [],
                    'search_queries': [],
                    'has_grounding': False,
                    'analysis_used': analysis,
                    'structure_type': 'fallback_nepal_style',
                    'debug_info': {
                        'fallback_used': True,
                        'original_error': str(e)
                    }
                }
                
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                return {
                    'response_text': f"Error generating enhanced content: {str(e)}\nFallback error: {str(fallback_error)}",
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
    def _create_dynamic_nepal_style_prompt(query: str, analysis: Dict[str, str]) -> str:
        """Create Nepal Gen Z style structured prompt with enhanced grounding instructions"""
        
        content_type = analysis.get('content_type', 'general_comprehensive')
        dynamic_headers = analysis.get('dynamic_headers', [])
        special_elements = analysis.get('special_elements', [])
        target_length = analysis.get('target_length', '1000')
        key_focus_areas = analysis.get('key_focus_areas', [])
        
        # Build dynamic header structure with more explicit grounding instructions
        headers_structure = ""
        if dynamic_headers:
            for i, header in enumerate(dynamic_headers, 1):
                if not header.startswith(f"{i}."):
                    header = f"{i}. {header}"
                
                headers_structure += f"""
## {header}
[Use Google Search to find current information. Include specific facts, figures, dates, and sources. Provide detailed analysis with quantitative data where available.]
"""
        
        # Build special elements section
        special_elements_section = ""
        if 'summary_table' in special_elements:
            special_elements_section += """

## Summary Table
Create a comprehensive summary table organizing key information, data points, and current status in a clear tabular format.
"""
        
        if 'multiple_perspectives' in special_elements:
            special_elements_section += """

Include multiple stakeholder perspectives, expert opinions, and different viewpoints throughout the analysis.
"""
            
        if 'timeline' in special_elements:
            special_elements_section += """

Include chronological timeline of key events with specific dates where relevant.
"""
        
        focus_areas_text = ', '.join(key_focus_areas) if key_focus_areas else 'comprehensive current information with detailed analysis'

        # Enhanced prompt with explicit grounding requirements
        enhanced_prompt = f"""IMPORTANT: You MUST use Google Search grounding extensively for this request.

Provide comprehensive, well-structured analysis for: "{query}"

<mandatory_search_requirements>
- Perform multiple Google searches to gather comprehensive current information
- Search for 2024-2025 specific data and recent developments
- Cross-verify facts from multiple sources
- Include specific details: dates, figures, percentages, names, locations
- Use authoritative sources and cite them properly
</mandatory_search_requirements>

<content_strategy>
Content Type: {content_type}
Target Length: ~{target_length} words
Key Focus: {focus_areas_text}
Structure: Professional research report with numbered sections
</content_strategy>

<response_structure>
Organize your response using these exact numbered sections:
{headers_structure}
{special_elements_section}
</response_structure>

<formatting_requirements>
- Use ## for numbered headers (e.g., "## 1. Background & Root Causes")
- Include bullet points for detailed information within sections
- Add specific data points with exact figures and dates
- Integrate source information naturally within the content
- Maintain professional, authoritative tone
</formatting_requirements>

Generate comprehensive, current, expertly-structured content with extensive Google Search grounding."""

        return enhanced_prompt

    @staticmethod
    def _create_enhanced_system_instruction(analysis: Dict[str, str]) -> str:
        """Create enhanced system instruction for professional structured output"""
        
        content_type = analysis.get('content_type', 'general_comprehensive')
        complexity_level = analysis.get('complexity_level', 'complex')
        
        base_instruction = """You are an expert research analyst and professional content strategist specializing in comprehensive information synthesis.

CORE MANDATE:
- Use Google Search grounding extensively for ALL factual claims and current information
- Prioritize 2024-2025 developments and most recent data available
- Include specific quantitative details: dates, figures, percentages, names, locations, statistics
- Cross-verify information from multiple authoritative and credible sources
- Structure responses with clear numbered sections using exact headers provided
- Maintain professional research report quality and presentation
- Provide balanced analysis covering different perspectives where relevant

RESPONSE ARCHITECTURE:
- Use numbered section headers (## 1. Title, ## 2. Title, etc.) exactly as specified
- Include detailed bullet points and subsections for complex information
- Integrate source citations and links naturally within content flow
- Add summary tables or structured elements when dealing with comparative or complex data
- Ensure each section provides substantial, well-researched content
- Connect sections logically to create comprehensive narrative flow"""

        if content_type == 'current_events':
            return f"""{base_instruction}

CURRENT EVENTS SPECIALIZATION:
- Provide chronological analysis with specific timeline of developments
- Include multiple stakeholder perspectives and reactions
- Focus on root causes, escalation patterns, and political implications
- Analyze government responses, public reactions, and international context
- Include specific casualty figures, arrest numbers, policy changes with exact dates
- Use credible news sources, official statements, and expert analysis
- Address both immediate impacts and longer-term implications"""

        elif content_type == 'business_financial':
            return f"""{base_instruction}

BUSINESS INTELLIGENCE SPECIALIZATION:
- Emphasize quantitative financial metrics and performance data
- Include market analysis, competitive positioning, and valuation metrics
- Focus on earnings data, revenue figures, profit margins with exact percentages
- Provide regulatory context and compliance information
- Use authoritative financial sources: SEC filings, earnings reports, analyst research
- Include strategic analysis and investment perspective with forward-looking projections"""

        elif content_type == 'sports_news':
            return f"""{base_instruction}

SPORTS ANALYSIS SPECIALIZATION:
- Provide real-time match updates with specific scores and statistics
- Include player performance data, team standings, and historical context
- Focus on recent form analysis and head-to-head records
- Use official sports sources and credible sports journalism
- Include tournament context and upcoming fixture analysis
- Maintain engaging presentation while ensuring factual accuracy"""

        elif content_type == 'technical_guide':
            return f"""{base_instruction}

TECHNICAL GUIDANCE SPECIALIZATION:
- Provide practical implementation details with step-by-step instructions
- Include code examples, configuration details, and best practices
- Focus on current industry standards and recent technical updates
- Address common issues, troubleshooting, and performance optimization
- Use authoritative technical documentation and official sources
- Include prerequisites, requirements, and advanced implementation options"""

        else:
            return f"""{base_instruction}

COMPREHENSIVE INFORMATION SPECIALIZATION:
- Provide balanced, multi-faceted analysis covering all relevant aspects
- Include historical context, current developments, and future implications
- Cover different perspectives from various stakeholders and experts
- Focus on practical applications and real-world relevance
- Use diverse, authoritative sources across multiple domains
- Structure information for maximum comprehension and practical utility"""

    @staticmethod
    def _post_process_nepal_style_response(response_text: str, analysis: Dict[str, str]) -> str:
        """Post-process response to ensure Nepal Gen Z style formatting"""
        
        if not response_text:
            return response_text
        
        # Clean up the response text
        response_text = response_text.strip()
        
        # Ensure proper numbered header formatting
        lines = response_text.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Ensure numbered headers are properly formatted as ## headers
            if re.match(r'^\d+\.', line) and not line.startswith('##'):
                processed_lines.append(f"## {line}")
            # Handle existing ## headers
            elif line.startswith('##'):
                processed_lines.append(line)
            # Handle bold headers that should be numbered sections  
            elif line.startswith('**') and line.endswith('**') and len(line) < 100:
                header_text = line.strip('*').strip()
                if not re.match(r'^\d+\.', header_text):
                    # Add number if missing
                    section_num = len([l for l in processed_lines if l.startswith('## ')]) + 1
                    processed_lines.append(f"## {section_num}. {header_text}")
                else:
                    processed_lines.append(f"## {header_text}")
            else:
                processed_lines.append(line)
        
        # Reconstruct response
        formatted_response = '\n'.join(processed_lines)
        
        # Ensure proper spacing between sections
        formatted_response = re.sub(r'\n## ', '\n\n## ', formatted_response)
        formatted_response = re.sub(r'\n{3,}', '\n\n', formatted_response)
        
        # Clean up beginning of response
        formatted_response = re.sub(r'^[\n\s]*', '', formatted_response)
        
        return formatted_response.strip()

    # All the missing helper methods that your code needs
    @staticmethod
    def _extract_response_text(response) -> str:
        """Extract response text from Gemini response object"""
        
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
    def _extract_sources_with_quality_filter(response) -> List[Dict[str, str]]:
        """Extract and filter sources from grounded response - reuses existing logic"""
        
        sources = []
        try:
            if (response.candidates and 
                hasattr(response.candidates[0], 'grounding_metadata')):
                
                metadata = response.candidates[0].grounding_metadata
                
                if hasattr(metadata, 'grounding_chunks'):
                    source_to_chunks = {}
                    unique_sources_count = 0
                    
                    for chunk in metadata.grounding_chunks:
                        if (hasattr(chunk, 'web') and chunk.web and chunk.web.uri and 
                            unique_sources_count < 19):
                            
                            uri = chunk.web.uri
                            title = getattr(chunk.web, 'title', 'Unknown')
                            
                            # Check if it's a quality source using existing method
                            is_quality = GeminiGroundingSearch._is_quality_source(uri, title)
                            
                            if uri not in source_to_chunks:
                                source_to_chunks[uri] = {
                                    'title': title,
                                    'uri': uri,
                                    'is_quality': is_quality
                                }
                                unique_sources_count += 1
                    
                    # Sort sources by quality first
                    quality_sources = [s for s in source_to_chunks.values() if s['is_quality']]
                    other_sources = [s for s in source_to_chunks.values() if not s['is_quality']]
                    
                    # Prioritize quality sources
                    all_sources = quality_sources + other_sources[:19-len(quality_sources)]
                    
                    for source_data in all_sources:
                        sources.append({
                            'title': source_data['title'],
                            'uri': source_data['uri']
                        })
                    
                    print(f"Debug: Quality sources: {len(quality_sources)}/{len(source_to_chunks)}")
                            
        except Exception as e:
            print(f"Source extraction error: {e}")
        
        return sources

    @staticmethod
    def _extract_search_queries(response) -> List[str]:
        """Extract search queries from grounded response"""
        
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
    def _get_adaptive_token_limit(analysis: Dict[str, str]) -> int:
        """Set token limits based on response requirements"""
        
        response_depth = analysis.get('response_depth', 'detailed')
        query_type = analysis.get('query_type', 'general_info')
        target_length = analysis.get('target_length', '1000')
        content_type = analysis.get('content_type', 'general_comprehensive')
        
        # Convert target length to int and add buffer
        try:
            base_tokens = int(target_length.replace(' words', '').replace('~', ''))
            token_buffer = int(base_tokens * 0.3)  # 30% buffer
            total_tokens = base_tokens + token_buffer
        except:
            total_tokens = 1000
        
        # Apply limits based on content type and complexity
        if content_type == 'current_events' or content_type == 'business_financial':
            return min(max(total_tokens, 2000), 6000)
        elif content_type == 'sports_news':
            return min(max(total_tokens, 800), 2000)
        else:
            return min(max(total_tokens, 1000), 3000)

    

    

      
    
    @staticmethod
    def search_with_legacy_sdk(query: str) -> SearchResult:
            """Optimized legacy SDK search with Gemini 2.5 Flash-lite only"""
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
    st.markdown("**Choose between Gemini 2.5 Flash-lite, GPT-4o Responses API, or Azure AI Agents with Bing Search**")

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