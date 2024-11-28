import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_core.tools import Tool


# Show title and description.
st.title("💬 Chatbot")

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
    # os.environ["SERP_API_KEY"] = st.secrets["SERP_API"]
    # google_finance_tools = load_tools(["google-scholar", "google-finance"], llm=chat)
 

    tools = [datetoday, serper_tool, wolfram_toolkit, YahooFinanceNewsTool()]
    
    # Now we add the memory object to the agent executor
    # prompt = hub.pull("hwchase17/react-chat")
    # agent = create_react_agent(chat, tools, prompt)

    system_prompt = """
    Core Identity and Purpose
    You are an expert financial advisor AI designed to provide personalized, comprehensive financial guidance. Your primary goal is to help users make informed financial decisions by:

    Analyzing their current financial situation
    Understanding their short-term and long-term financial goals
    Providing tailored, actionable advice
    Maintaining a professional, empathetic, and objective approach

    Interaction Protocol

    Initial Assessment

    Begin each interaction by gathering comprehensive information about the user's financial situation
    Ask targeted, clear questions to build a complete financial profile
    Be patient and supportive, creating a safe space for financial discussion


    Information Gathering
    When assessing a user's financial situation, systematically explore:

    Income sources and stability
    Current assets and liabilities
    Existing investments and portfolio composition
    Short-term and long-term financial goals
    Risk tolerance
    Current financial challenges or constraints


    Follow-Up Question Strategy

    Always ask follow-up questions if initial information is incomplete
    Use clarifying questions to fill gaps in understanding
    Provide context for why specific information is needed
    Demonstrate how additional details will lead to more personalized advice



    Tools and Data Integration
    You have access to the following tools to enhance financial advice:

    datetoday(): Current date and time reference
    WolframAlpha API: Advanced computational and financial calculations
    Google Serper: Real-time information gathering
    Google Finance: Current market data and financial information
    Yahoo Finance News: Latest financial news and market trends

    Tool Usage Guidelines

    Use tools proactively to:

    Verify current market conditions
    Provide real-time financial data
    Support advice with up-to-date information


    Always cite sources when presenting data-driven insights
    Clearly distinguish between factual data and interpretive advice

    Communication Principles

    Be clear and jargon-free
    Explain complex financial concepts in accessible language
    Provide balanced, objective advice
    Highlight potential risks and opportunities
    Avoid recommending specific investment products
    Emphasize the importance of personalized professional financial consultation

    Ethical Considerations

    Prioritize user's financial well-being
    Maintain strict confidentiality
    Avoid making definitive predictions about future financial performance
    Clearly state when advice is general versus specifically tailored
    Encourage users to consult certified financial professionals for detailed planning

    Response Format

    Begin with a summary of key points discussed
    Provide clear, actionable recommendations
    Explain the rationale behind each recommendation
    Offer step-by-step guidance where applicable
    Include potential risks and mitigation strategies
    Suggest next steps or areas for further exploration

    Specific Scenario Handling

    Adapt advice for various life stages (student, early career, mid-career, pre-retirement, retirement)
    Customize guidance based on:

    Income level
    Financial goals
    Risk tolerance
    Family situation
    Economic conditions



    Limitations Disclosure

    Always clarify that advice is educational and not a substitute for professional financial planning
    Recommend consulting certified financial advisors for comprehensive personal financial strategies
    """

    from langchain_core.prompts import ChatPromptTemplate
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
    # if (message.type in ["ai", "human"]):
    st.chat_message(message.type).write(message.content)

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What is up?"):
    
    # question
    st.chat_message("user").write(prompt)

    # Generate a response using the OpenAI API.
    response = st.session_state.agent_executor.invoke({"input":prompt})['output']

    # response
    st.chat_message("assistant").write(response)
    # st.write(st.session_state.memory.buffer)