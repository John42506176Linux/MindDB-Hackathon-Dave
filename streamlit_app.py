import streamlit as st
import rag_helper as rh
import tempfile
from streamlit_chat import message
from whispher import whisper_stt
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_together.embeddings import TogetherEmbeddings
from langchain_openai import ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain.tools.retriever import create_retriever_tool
import os
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings, play
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_groq import ChatGroq


client = MongoClient(os.environ['MONGODB_URI'])
# Define collection and index name
db_name = "langchain_db"
collection_name = "test2"
atlas_collection = client[db_name][collection_name]
vector_search_index = "vector_index"
global file_name
st.title("Dave- An AI Software Architect in your pocket")
uploaded_file = st.sidebar.file_uploader("Upload your Requirements Files", type="txt")
labs_client = ElevenLabs(
    api_key=os.getenv('ELEVENLABS_API_KEY'),  # Defaults to ELEVEN_API_KEY
)
khan_voice = Voice(
    voice_id=os.getenv('KHAN_VOICE_ID'),
    settings=VoiceSettings(
        stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
)

if not uploaded_file:
    st.error("Please upload brainstorming/chat transcript to continue!")
    st.stop()
else:
    print("File uploaded")
    st.button('Start')
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
        file_name = tmp_file_path
        with st.spinner("Processing the file"):
            rh.embed_pdf(tmp_file_path)
            answer = rh.generate_design_document(tmp_file_path)
        st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": answer})
        def generate_diagram():
            rh.generate_architecture_diagram_code(answer)
        st.button('Generate Architecture Diagram',on_click=generate_diagram)

def main():
    # Initialize chat history
    
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ['MONGODB_URI'],
        db_name + "." + collection_name,
        TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval"),
        index_name=vector_search_index,
    )
    retriever = vector_search.as_retriever(
        search_type = "similarity",
        search_kwargs = {
            "k": 10,
            "score_threshold": 0.75,
            "pre_filter": { "source": { "$eq": file_name  } }
        }
    )
    tool = create_retriever_tool(
        retriever,
        "get_initial_conversation_transcript",
        "Searches and returns information from the transcript of the original brainstorming meeting.",
    )
    tools = [tool]

    memory_key = "history"
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                You are an experienced software architect and you have been asked to design a software architecture for a new project. 
                You've currently designed the following design document:{answer}
                This design document was created from the brainstorm seesion you had with your team. You can access that with the tool get_initial_conversation_transcript.
                However, your superiors are speaking with you and want to know more about this design document. They may ask you to change it, so use your database of the previous discussion and the current discussion in order to proceed.
                Make sure to be friendly.
    """,
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent  = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
                                    return_intermediate_steps=True)
    # Function for conversational chat
    def conversational_chat(query):
        result = agent_executor({"input": query})
        return result["output"]
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Get response from conversational chat 
        with st.chat_message("bot"):
            try:
                output = conversational_chat(prompt)
                st.markdown(output)
            except:
                st.markdown("I am sorry, I did not understand that. Can you please rephrase?")
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    
    if voice_prompt := whisper_stt(start_prompt="Start recording", stop_prompt="Stop recording"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(voice_prompt)
           
            # Display AI response in chat message container
        with st.chat_message("bot"):
            try:
                output = conversational_chat(voice_prompt)
                st.markdown(output)
                audio = labs_client.generate(
                    text=output,
                    voice="Rachel",
                    model="eleven_multilingual_v2"
                )
                play(audio)
            except:
                st.markdown("I am sorry, I did not understand that. Can you please rephrase?")
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": voice_prompt})
    

if __name__ == "__main__":
    main()