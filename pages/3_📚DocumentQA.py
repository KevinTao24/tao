import streamlit as st
from langchain_community.chat_models import ChatZhipuAI


def initialize_chat():

    # Streamlit file uploader widget
    uploaded_files = st.file_uploader(
        "choose .pdf/.txt file",
        accept_multiple_files=True,
        type=["pdf", "text", "txt"],
        key="a",
    )

    # upload the files to AI21 RAG Engine library
    for uploaded_file in uploaded_files:
        client.library.files.create(file_path=uploaded_file, labels=label)

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
    st.set_page_config("DocumentQA", page_icon="🚀")
    st.title("📚 DocumentQA")
    st.caption("🚀 A Streamlit DocumentQA powered by AIGC")

    initialize_chat()
    chat_model = setup_chat_model()
    display_chat()
    handle_user_input(chat_model)


if __name__ == "__main__":
    main()
