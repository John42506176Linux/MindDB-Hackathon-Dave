import streamlit as st
from streamlit_chat import message

st.title("Dave- An AI Software Architect in your pocket")
uploaded_file = st.sidebar.file_uploader("Upload your Requirements Files", type="txt")


if not uploaded_file:
    st.error("Please upload brainstorming/chat transcript to continue!")
    st.stop()
else:
    print("File uploaded")
    st.button('Start')

def main():
    # Initialize chat history
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
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

if __name__ == "__main__":
    main()
