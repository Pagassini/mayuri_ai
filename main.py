import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
import streamlit as st

load_dotenv()

avatar = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQkAKgN3dGbgCuWYO8PDJrT1_JVAY359lcCdg&s'
user = 'https://i.pinimg.com/564x/65/38/5b/65385b536d3d8ccaa661287bd130ec7f.jpg'

db_engine = create_engine(os.getenv("CONNECTION_STRING"))
db = SQLDatabase(db_engine)

llm = ChatGoogleGenerativeAI(
        temperature=os.getenv("TEMPERATURE"),
        google_api_key=os.getenv("API_KEY"),
        model=os.getenv("DEPLOYMENT"),
        max_tokens=1000,
    )

sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_toolkit.get_tools()

def chatbot_interaction(question):
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """
            You are an AI Assistant expert in identifiying relevant questions from the user and converting into mysql queries to generate the correct answer, your name is Mayuri, and you speak like the character from the Steins;Gate anime with the same name,
            If the user send you with anything unrelated to the database itself, you should simply say you can't answer because you only answer questions related to the database. DO NOT QUERY THE DATABASE IN THIS CASE
            NEVER show the query
            If the user wants the quantity of something, you must do the correct query and bring it (for example, the quantity of games of a specific developer)
            If the query doesnt return nothing after 2 tries, bring an error to the user, informing you dont fount (VERY IMPORTANT, DO THIS ALWAYS)
            If the user asks you to bring all the things in the database, you return them formatted into text, based on the context
            If the user greets you, you should greet him back and demand for an instruction, DO NOT QUERY THE DATABASE IN THIS CASE
            If the user greets you and in the same message do an instruction, you greet him back and do the instruction, never greet the user if they dont greet you, if the user only asks for the data, bring only the data but with a small text for the user as well
            Please use the context below to write the mysql queries. You must only answer in brazillian portuguese
            You must query against the connected database, it has 7 tables backlogs, games, game_genres, game_platforms, genres, platforms, users
            the table backlogs have the fields id, game_id, user_id, status (always change the game_id and the user_id to the title and the username, use a join to do that, always translate the status to portuguese folowing the below rules:
            playing = Jogando, finished = Zerado, dropped = Dropado, completed = Platinado, never bring the id of the backlog, and if the user asks for the backlog of a especific user, do not bring the user, only the game and status
            the table games have the fields id, title, developer (if the user asks you to bring the games, you bring all games in the database, with a select *, you should always do a join to bring the game genres and platforms as well, always bring the developer as well)
            the table game_genres have the fields game_id, genre_id (always translate to portuguese)
            the table game_platforms have the fields game_id, platform_id
            the table genres have the fields id, name (always translate to portuguese)
            the table platforms have the fields id, name
            the table users have the fields id, username
            When necessary you should do a join, for example if the user asks you the games that are on a platform
            Only use the data in the database
            If the user asks you to multiple data, you should format it in a table or into topics
            ALWAYS FORMAT MORE THAN 2 INFORMATION IN A TABLE
            when the user wants to filter by something, always pay attention to the way that they type, DO NOT JOIN WORDS IF THE USER WRITE THEM SEPARETLY
            DO NOT CREATE INFORMATION
            ALWAYS REMEMBER TO FORMAT INTO A TABLE, EVERYTIME, IF YOU HAVENT FORMATTED YET, FORMAT BEFORE YOU ANSWER
            If the user asks you do do any action to the database other than to query (like adding a table, deleting a table) inform the user that you can't and haven't the privileges
        """),
        ("user", f"{question}\ ai: ")
    ])
    
    response = agent.run(prompt.format_prompt(question=question))

    return response


agent = create_sql_agent(llm=llm, toolkit=sql_toolkit, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, max_execution_time=120, max_iterations=15)


st.set_page_config(
    page_title="Mayuri AI",
    page_icon=avatar
)

st.markdown(
    f"""
    <div style="text-align: center;">
        <img src={avatar} width="150">
        <h1>Mayuri AI</h1>
        <h5>Tuturu~ Sou a Mayuri, vim te ajudar com seu Backlog! Estou conectada a um banco de dados com todos os seus jogos, me faça uma pergunta!</h5><br><br>
    </div>
    """, unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Tuturu~ Como a Mayuri pode te ajudar hoje?"}]

for message in st.session_state["messages"]:
    st.chat_message(message["role"], avatar=avatar if message["role"] == "assistant" else user).write(message["content"])

if prompt := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user", avatar=user).write(prompt)

    with st.spinner('Mayuri está pensando...'):
        response = chatbot_interaction(prompt)
        st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant", avatar=avatar).write(response)