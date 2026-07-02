from langchain_community.tools import ShellTool
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chat_models import ChatOllama
from langchain.agents import create_json_chat_agent, AgentExecutor
from langchain import hub

# 1. Setup Local LLM (Ollama)
llm = ChatOllama(model="llama3.1", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Pre-built RAG: Load and Index your Markdown files in 3 lines
loader = DirectoryLoader(
    "./my_docs", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
)
docs = loader.load()
vector_store = Chroma.from_documents(docs, embeddings)
retriever_tool = vector_store.as_retriever()  # This is now a tool for the agent!

# 3. Pre-built Action: Grab LangChain's built-in Shell Tool
shell_tool = ShellTool()

# 4. Combine into an Agent
tools = [retriever_tool, shell_tool]
prompt = hub.pull("hwchase17/react-json")  # Pre-built reasoning prompt template

agent = create_json_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Run it!
agent_executor.invoke(
    {
        "input": "Look at my deployment docs and check if the docker container is running."
    }
)
