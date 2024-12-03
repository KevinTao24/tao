import streamlit as st
from langchain_community.chat_models import ChatZhipuAI


def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]


def setup_chat_model():
    zhipu_api_key = st.secrets.get("general", {}).get("zhipu_api_key")
    if not zhipu_api_key:
        st.info("Please add your Zhipu AI API key to continue...")
        st.stop()

    return ChatZhipuAI(
        api_key=zhipu_api_key, model="glm-4-plus", temperature=0.99, streaming=True
    )


def display_chat():
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])


def handle_user_input(chat_model: ChatZhipuAI):
    if prompt := st.chat_input(placeholder="Ask me anything..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        msg = st.chat_message("assistant").write_stream(
            chat_model.stream(st.session_state["messages"])
        )

        st.session_state["messages"].append({"role": "assistant", "content": msg})


def main():
    st.title("🤖 Chatbot")
    st.caption("🚀 A Streamlit chatbot powered by Zhipu AI")

    initialize_chat()
    chat_model = setup_chat_model()
    display_chat()
    handle_user_input(chat_model)


if __name__ == "__main__":
    main()
