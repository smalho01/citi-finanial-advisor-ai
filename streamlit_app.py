import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
import re
from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain import hub
# from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
import mdtex2html


# Updated Custom CSS for Citi Bank branding
st.markdown("""
<style>
    /* Citi Bank Color Palette */
    :root {
        --citi-blue: #003CA6;  /* Primary Citi blue */
        --citi-light-blue: #00A4E4;  /* Accent blue */
        --citi-white: #FFFFFF;
        --citi-gray: #F5F5F5;
        --citi-button-hover: #002A85;  /* Darker blue for hover state */
    }

    /* App Container Styling */
    .stApp {
        background-color: var(--citi-gray);
        color: var(--citi-blue);
    }
            
    /* Target Streamlit header */
    .stApp header {
        background-color: #c4c4c4 !important;
    }
            
    /* Target Streamlit footer */
    .stBottom, .st-emotion-cache-hzygls, .st-emotion-cache-arzcut, .st-emotion-cache-0, .st-emotion-cache-1wmy9hl, .st-emotion-cache-1n76uvr, .stElementContainer {
        background-color: #c4c4c4 !important;
    }

    /* Enhanced Header Styling */
    .stTitle {
        color: var(--citi-blue);
        font-weight: bold;
        text-align: center;
        padding: 15px 0;
        margin-bottom: 20px;
        border-bottom: 2px solid var(--citi-blue);
    }

    /* Google Form Button Styling */
    .google-form-button {
        background-color: var(--citi-blue);
        color: var(--citi-white);
        border: 2px solid var(--citi-light-blue);
        border-radius: 10px;
        padding: 10px 20px;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s ease;
        margin-left: 20px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .google-form-button:hover {
        background-color: var(--citi-button-hover);
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }

    /* Existing styles remain the same */
    .stTextInput > div > div > input {
        border: 2px solid var(--citi-blue);
        border-radius: 8px;
        padding: 10px;
    }

    .user-message {
        background-color: var(--citi-light-blue);
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }

    .assistant-message {
        background-color: white;
        border: 1px solid var(--citi-blue);
        color: var(--citi-blue);
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }

    /* Escape math formatting */
    .stMarkdown p {
        white-space: pre-wrap;
        word-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

# In your Streamlit app, you can add the button like this
st.markdown("""
    <a href="https://docs.google.com/forms/d/e/1FAIpQLSeg4V_DfpXpj24rgBtiLUVOJ0aaQRwBCkh0QsNcnmDzSPG18g/viewform?usp=sf_link" class="google-form-button" target="_blank">
        Provide Feedback
    </a>
""", unsafe_allow_html=True)

def escape_math_symbols(text):
   
    formatted_result = text.replace('$', '&#36;') 
    formatted_result = mdtex2html.convert(formatted_result)

    return formatted_result

# Show title and description.
st.title("💬 Citi Bank Financial Assistant")

### Important part.
# Create a session state variable to flag whether the app has been initialized.
# This code will only be run first time the app is loaded.
if "memory" not in st.session_state: ### IMPORTANT.
    model_type="gpt-4o"

    # initialize the momory
    max_number_of_exchanges = 15
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=max_number_of_exchanges, return_messages=True) ### IMPORTANT to use st.session_state.memory.

    # LLM
    os.environ["OPENAI_API_KEY"] = st.secrets["OpenAI_API_KEY"]
    chat = ChatOpenAI(model=model_type)

    # tools
    from langchain.agents import tool
    from datetime import date
    @tool
    def datetoday(dummy: str) -> str:
        """Returns today's date, use this for any \
        questions that need today's date to be answered. \
        This tool returns a string with today's date.""" #This is the desciption the agent uses to determine whether to use the time tool.
        return "Today is " + str(date.today())
    
    # Setting up the Serper tool
    os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
    search = GoogleSerperAPIWrapper()

    serper_tool = Tool(
        name="GoogleSerper",
        func=search.run,
        description="Useful for when you need to look up some information on the internet.",
    )

    # Wolfram Alpha 
    os.environ["WOLFRAM_ALPHA_APPID"] = st.secrets["WOLFRAM_ALPHA_APPID"]
    wolfram = WolframAlphaAPIWrapper()
    wolfram_toolkit = Tool(
        name="WolframAlpha",
        func=wolfram.run,
        description="Use WolframAlpha for complex mathematical or scientific queries."
    )

    # # Google Finance 
    # os.environ["SERPAPI_API_KEY"] = st.secrets["SERP_API"]
    # google_finance_tools = load_tools(["google-finance"])

    # Alpha Vantage Tool 
    os.environ["ALPHAVANTAGE_API_KEY"] = st.secrets["ALPHAVANTAGE_API_KEY"]

    # Initialize Alpha Vantage API Wrapper
    alpha_vantage = AlphaVantageAPIWrapper()

    # Tool for currency exchange rates
    exchange_rate_tool = Tool(
        name="CurrencyExchange",
        description="Get the exchange rate between two currencies. Provide 'from_currency' and 'to_currency' as inputs.",
        func=(lambda query: alpha_vantage._get_exchange_rate(*query.split(",")))
    )

    # Tool for daily time series
    time_series_daily_tool = Tool(
        name="TimeSeriesDaily",
        description="Get daily time series data for a stock. Provide the stock symbol as input.",
        func=alpha_vantage._get_time_series_daily
    )

    # Tool for weekly time series
    time_series_weekly_tool = Tool(
        name="TimeSeriesWeekly",
        description="Get weekly time series data for a stock. Provide the stock symbol as input.",
        func=alpha_vantage._get_time_series_weekly
    )

    # Tool for stock quote
    quote_tool = Tool(
        name="StockQuote",
        description="Get the latest stock price and volume information. Provide the stock symbol as input.",
        func=alpha_vantage._get_quote_endpoint
    )

    # Tool for symbol search
    symbol_search_tool = Tool(
        name="SymbolSearch",
        description="Search for stock symbols. Provide a partial or full name of a company as input.",
        func=alpha_vantage.search_symbols
    )

    # Tool for market news sentiment
    market_news_tool = Tool(
        name="MarketNewsSentiment",
        description="Get live and historical market news sentiment for a given asset. Provide the stock symbol as input.",
        func=alpha_vantage._get_market_news_sentiment
    )

    # Tool for top gainers, losers, and most active stocks
    top_gainers_losers_tool = Tool(
        name="TopGainersAndLosers",
        description="Get the top 20 gainers, losers, and most active stocks in the US market. No input needed.",
        func=(lambda _: alpha_vantage._get_top_gainers_losers())
    )


    alpha_vantage_tools = [
    exchange_rate_tool,
    time_series_daily_tool,
    time_series_weekly_tool,
    quote_tool,
    symbol_search_tool,
    market_news_tool,
    top_gainers_losers_tool,
    ]

    tools = [datetoday, serper_tool, wolfram_toolkit, YahooFinanceNewsTool()] + alpha_vantage_tools
    
    # Now we add the memory object to the agent executor
    # prompt = hub.pull("hwchase17/react-chat")
    # agent = create_react_agent(chat, tools, prompt)

    system_prompt = """
    
    ## Core Identity

    You are a professional financial advisor AI designed to provide personalized, actionable financial guidance. Your goal is to help users make informed financial decisions by understanding their unique situation, goals, and challenges.
    Interaction Protocol

    ## Interaction Guidelines

    Communicate in a conversational, approachable manner
    Use concise, clear language
    Ask follow-up questions to gather comprehensive information
    Break down complex financial concepts into easily understandable insights
    Provide personalized advice tailored to the user's specific circumstances

    ## Initial Assessment

    Begin each interaction by gathering comprehensive information about the user's financial situation
    Ask targeted, clear questions to build a complete financial profile
    Be patient and supportive, creating a safe space for financial discussion

    ## Communication Strategy

    Start with active listening
    Ask clarifying questions to understand the user's financial context
    Provide targeted, actionable advice
    Offer step-by-step guidance
    Use tools to validate and enhance recommendations


    ## Information Gathering
    When assessing a user's financial situation, systematically explore:

    Income sources and stability
    Current assets and liabilities
    Existing investments and portfolio composition
    Short-term and long-term financial goals
    Risk tolerance
    Current financial challenges or constraints

    ## Follow-Up Question Strategy

    Always ask follow-up questions if initial information is incomplete
    Use clarifying questions to fill gaps in understanding
    Provide context for why specific information is needed
    Demonstrate how additional details will lead to more personalized advice

    ## Tools and Data Integration
    
    You have access to the following tools to enhance financial advice:

    datetoday(): Current date and time reference
    WolframAlpha API: Advanced computational and financial calculations
    Google Serper: Real-time information gathering
    Yahoo Finance News: Latest financial news and market trends
    Currency Exchange Rate Tool: Get exchange rates between currencies
    Stock Time Series Tools: Daily/Weekly stock price data
    Stock Quote Tool: Latest stock price and volume information
    Symbol Search Tool: Find stock symbols for companies
    Market News Sentiment Tool: Live and historical market news sentiment
    Market Overview Tools: Top gainers and losers most active stocks in US market

    ## Tool Usage Guidelines

    Use tools proactively to:

    Verify current market conditions
    Provide real-time financial data
    Support advice with up-to-date information

    Always cite sources when presenting data-driven insights
    Clearly distinguish between factual data and interpretive advice

    ## Communication Principles

    Be clear and jargon-free
    Explain complex financial concepts in accessible language
    Provide balanced, objective advice
    Highlight potential risks and opportunities

    ## Response Format

    Final output is rendered in markdown so make sure any math formulas are in a friendly format
    Use short, digestible messages
    Ask follow-up questions when information is incomplete
    Provide clear, actionable recommendations
    Use real-time data to support advice
    Explain reasoning behind recommendations

    ## Specific Scenario Handling

    Adapt advice for various life stages (student, early career, mid-career, pre-retirement, retirement)
    Customize guidance based on:

    Income level
    Financial goals
    Risk tolerance
    Family situation
    Economic conditions
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(chat, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools,  memory=st.session_state.memory, verbose= True)  # ### IMPORTANT to use st.session_state.memory and st.session_state.agent_executor.

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.memory.buffer:
    safe_content = escape_math_symbols(message.content)
    if message.type == "human":
        st.markdown(f'<div class="user-message">{safe_content}</div>', unsafe_allow_html=True)
    elif message.type == "ai":
        st.markdown(f'<div class="assistant-message">{safe_content}</div>', unsafe_allow_html=True)


# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What financial advice do you need today?"):
    
    # question
    safe_prompt = escape_math_symbols(prompt)
    st.markdown(f'<div class="user-message">{safe_prompt}</div>', unsafe_allow_html=True)

    # Generate a response using the OpenAI API.
    response = st.session_state.agent_executor.invoke({"input":prompt})['output']

    # response
    safe_response = escape_math_symbols(response)
    st.markdown(f'<div class="assistant-message">{safe_response}</div>', unsafe_allow_html=True)
    # st.write(st.session_state.memory.buffer)