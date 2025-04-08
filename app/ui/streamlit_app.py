import streamlit as st
import requests
import json
import pandas as pd
import time
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="Financial Research Copilot",
    page_icon="📊",
    layout="wide"
)

# API endpoint
API_URL = os.environ.get("API_URL", "http://localhost:8000/api/v1")

# Helper functions
def call_api(endpoint, method="get", data=None, params=None, files=None):
    """
    Call API endpoint
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        data: Request data
        params: Query parameters
        files: Files to upload
        
    Returns:
        API response
    """
    url = f"{API_URL}/{endpoint}"
    
    try:
        if method.lower() == "get":
            response = requests.get(url, params=params)
        elif method.lower() == "post":
            if files:
                response = requests.post(url, data=data, params=params, files=files)
            else:
                response = requests.post(url, json=data, params=params)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
            
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}

# Sidebar
st.sidebar.title("Financial Research Copilot")
st.sidebar.write("Powered by RAG and LLMs")

# Navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Company Research", "Document Upload", "Query Interface", "Sentiment Analysis"]
)

# Home page
if page == "Home":
    st.title("Financial Research Copilot")
    st.subheader("An AI-powered investment research assistant")
    
    st.write("""
    This application combines vector databases and large language models to analyze financial reports, 
    SEC filings, and market data in real-time, helping you make better investment decisions.
    
    ### Features
    
    - **Document Ingestion**: Upload financial documents or ingest company SEC filings
    - **Company Research**: Generate comprehensive research reports on companies
    - **Natural Language Queries**: Ask questions about companies in plain English
    - **Sentiment Analysis**: Analyze market sentiment for companies
    """)
    
    st.info("Use the sidebar to navigate between different features.")
    
    # Quick query example
    st.subheader("Quick Query")
    with st.form("quick_query_form"):
        query = st.text_input("Ask a financial question:")
        ticker = st.text_input("Company Ticker (optional):")
        submitted = st.form_submit_button("Submit")
        
        if submitted and query:
            with st.spinner("Analyzing..."):
                request_data = {"query": query}
                if ticker:
                    request_data["ticker"] = ticker
                
                response = call_api("query", method="post", data=request_data)
                
                if "error" in response:
                    st.error(response["error"])
                else:
                    st.write("### Answer")
                    st.write(response.get("answer", "No answer found"))
                    
                    if response.get("sources"):
                        st.write("### Sources")
                        for source in response.get("sources", []):
                            st.write(f"- {source.get('type', '')} | {source.get('source', '')}")

# Company Research page
elif page == "Company Research":
    st.title("Company Research")
    
    with st.form("company_form"):
        ticker = st.text_input("Company Ticker Symbol:")
        st.write("Select topics to include in the research report:")
        
        col1, col2 = st.columns(2)
        with col1:
            topic1 = st.checkbox("Financial Performance", value=True)
            topic2 = st.checkbox("Business Overview", value=True)
            topic3 = st.checkbox("Competitive Analysis", value=True)
        with col2:
            topic4 = st.checkbox("Risks and Challenges", value=True)
            topic5 = st.checkbox("Future Outlook", value=True)
            topic6 = st.checkbox("Valuation", value=False)
        
        time_period = st.selectbox(
            "Time Period:",
            ["Last Quarter", "Last Year", "Last 3 Years", "Last 5 Years", "All Available Data"]
        )
        
        ingest_data = st.checkbox("Ingest latest company data first (recommended)", value=True)
        submitted = st.form_submit_button("Generate Research Report")
    
    if submitted and ticker:
        ticker = ticker.upper()
        
        # Build topics list
        topics = []
        if topic1: topics.append("Financial Performance")
        if topic2: topics.append("Business Overview")
        if topic3: topics.append("Competitive Analysis")
        if topic4: topics.append("Risks and Challenges")
        if topic5: topics.append("Future Outlook")
        if topic6: topics.append("Valuation")
        
        if ingest_data:
            # Ingest company data first
            with st.spinner(f"Ingesting latest data for {ticker}..."):
                ingest_response = call_api(
                    f"companies/{ticker}/ingest",
                    method="post",
                    params={"filing_types": ["10-K", "10-Q", "8-K"], "limit_per_type": 3}
                )
                
                if "error" in ingest_response:
                    st.error(f"Error ingesting data: {ingest_response['error']}")
                else:
                    st.success(f"Started data ingestion for {ticker}")
                    # Wait for a moment to allow some data to be ingested
                    time.sleep(5)
        
        # Generate research report
        with st.spinner(f"Generating research report for {ticker}..."):
            research_request = {
                "ticker": ticker,
                "topics": topics,
                "time_period": time_period
            }
            
            response = call_api("research", method="post", data=research_request)
            
            if "error" in response:
                st.error(response["error"])
            else:
                report_id = response.get("report_id")
                st.success(f"Started research report generation for {ticker}")
                st.info("Research report generation is running in the background. This may take a few minutes.")
                
                # Create expander for checking status
                with st.expander("Check report status"):
                    check_status = st.button("Check Status")
                    
                    if check_status and report_id:
                        report_response = call_api(f"research/{report_id}")
                        
                        if "error" in report_response:
                            st.error(report_response["error"])
                        else:
                            status = report_response.get("status", "")
                            st.write(f"Status: {status}")
                            
                            if status == "completed":
                                st.success("Report is ready!")
                                
                                # Display report
                                st.write("### Executive Summary")
                                st.write(report_response.get("summary", ""))
                                
                                for section in report_response.get("sections", []):
                                    st.write(f"### {section.get('title', '')}")
                                    st.write(section.get("content", ""))
                                
                                st.write("### Sources")
                                for source in report_response.get("sources", []):
                                    st.write(f"- {source.get('type', '')} | {source.get('source', '')}")
                            
                            elif status == "error":
                                st.error(f"Error generating report: {report_response.get('error_message', 'Unknown error')}")
                            else:
                                st.info(f"Report is still being generated. Status: {status}")

# Document Upload page
elif page == "Document Upload":
    st.title("Document Upload")
    st.write("Upload financial documents to include in the research database.")
    
    with st.form("upload_form"):
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"])
        ticker = st.text_input("Associated Company Ticker (optional):")
        content_type = st.selectbox(
            "Document Type:",
            ["Financial Report", "Research Note", "News Article", "Transcript", "Other"]
        )
        submitted = st.form_submit_button("Upload")
    
    if submitted and uploaded_file:
        with st.spinner("Uploading and processing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            params = {"content_type": content_type.lower().replace(" ", "_")}
            
            if ticker:
                params["ticker"] = ticker.upper()
            
            response = call_api("documents/upload", method="post", params=params, files=files)
            
            if "error" in response:
                st.error(response["error"])
            else:
                st.success("Document uploaded and processing started")
                st.json(response)
    
    # Also allow ingesting company data directly
    st.subheader("Ingest Company Data")
    st.write("Ingest SEC filings, news, and financial data for a company.")
    
    with st.form("ingest_form"):
        ingest_ticker = st.text_input("Company Ticker Symbol:")
        filing_types = st.multiselect(
            "Filing Types:",
            ["10-K", "10-Q", "8-K", "DEF 14A", "S-1"],
            default=["10-K", "10-Q", "8-K"]
        )
        limit = st.slider("Limit per filing type:", 1, 10, 3)
        include_news = st.checkbox("Include News", value=True)
        include_financials = st.checkbox("Include Financial Data", value=True)
        
        ingest_submitted = st.form_submit_button("Ingest Data")
    
    if ingest_submitted and ingest_ticker:
        ingest_ticker = ingest_ticker.upper()
        
        with st.spinner(f"Ingesting data for {ingest_ticker}..."):
            params = {
                "filing_types": filing_types,
                "limit_per_type": limit,
                "include_news": include_news,
                "include_financials": include_financials
            }
            
            response = call_api(f"companies/{ingest_ticker}/ingest", method="post", params=params)
            
            if "error" in response:
                st.error(response["error"])
            else:
                st.success(f"Started data ingestion for {ingest_ticker}")
                st.json(response)

# Query Interface page
elif page == "Query Interface":
    st.title("Query Interface")
    st.write("Ask questions about companies and financial data.")
    
    with st.form("query_form"):
        query = st.text_area("Your Question:")
        ticker = st.text_input("Company Ticker (optional):")
        
        col1, col2 = st.columns(2)
        with col1:
            content_sec = st.checkbox("Search SEC Filings", value=True)
            content_news = st.checkbox("Search News", value=True)
        with col2:
            content_fin = st.checkbox("Search Financial Data", value=True)
            expand_query = st.checkbox("Expand Query with Financial Terms", value=True)
        
        submitted = st.form_submit_button("Submit Query")
    
    if submitted and query:
        with st.spinner("Processing query..."):
            # Build content types list
            content_types = []
            if content_sec: content_types.append("sec_filing")
            if content_news: content_types.append("news")
            if content_fin: content_types.append("financial_data")
            
            request_data = {
                "query": query,
                "expand_query": expand_query,
                "content_types": content_types
            }
            
            if ticker:
                request_data["ticker"] = ticker.upper()
            
            response = call_api("query", method="post", data=request_data)
            
            if "error" in response:
                st.error(response["error"])
            else:
                st.write("### Answer")
                st.write(response.get("answer", "No answer found"))
                
                if response.get("expanded_query"):
                    with st.expander("Expanded Query"):
                        st.write(response.get("expanded_query"))
                
                if response.get("sources"):
                    st.write("### Sources")
                    for source in response.get("sources", []):
                        st.write(f"- {source.get('type', '')} | {source.get('source', '')}")

# Sentiment Analysis page
elif page == "Sentiment Analysis":
    st.title("Sentiment Analysis")
    st.write("Analyze market sentiment for companies.")
    
    with st.form("sentiment_form"):
        ticker = st.text_input("Company Ticker:")
        days = st.slider("Number of days to analyze:", 7, 90, 30)
        submitted = st.form_submit_button("Analyze Sentiment")
    
    if submitted and ticker:
        ticker = ticker.upper()
        
        with st.spinner(f"Analyzing sentiment for {ticker}..."):
            response = call_api(f"sentiment/{ticker}", params={"days": days})
            
            if "error" in response:
                st.error(response["error"])
            else:
                sentiment_score = response.get("average_sentiment", 0)
                sentiment_category = response.get("sentiment_category", "neutral")
                
                # Display sentiment score with color
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Sentiment", f"{sentiment_score:.2f}")
                with col2:
                    color = "green" if sentiment_category == "positive" else "red" if sentiment_category == "negative" else "blue"
                    st.markdown(f"<h3 style='color: {color};'>{sentiment_category.capitalize()}</h3>", unsafe_allow_html=True)
                
                # Display sentiment distribution
                distribution = response.get("sentiment_distribution", {})
                
                if distribution:
                    st.subheader("Sentiment Distribution")
                    dist_data = pd.DataFrame({
                        "Category": ["Positive", "Neutral", "Negative"],
                        "Percentage": [
                            distribution.get("positive", 0),
                            distribution.get("neutral", 0),
                            distribution.get("negative", 0)
                        ]
                    })
                    
                    st.bar_chart(dist_data.set_index("Category"))
                
                # Display detailed results
                st.subheader("Detailed Results")
                
                results = response.get("detailed_results", [])
                if results:
                    for result in results:
                        sentiment = result.get("sentiment_score", 0)
                        category = result.get("sentiment_category", "neutral")
                        color = "green" if category == "positive" else "red" if category == "negative" else "blue"
                        
                        st.markdown(f"""
                        <div style='padding: 10px; border-left: 5px solid {color}; margin-bottom: 10px;'>
                            <strong>{result.get('headline', '')}</strong><br>
                            Source: {result.get('source', '')} | Date: {result.get('date', '')}<br>
                            Sentiment: <span style='color: {color};'>{sentiment:.2f} ({category})</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No detailed results available.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Financial Research Copilot with RAG")
st.sidebar.caption(f"© {datetime.now().year}")