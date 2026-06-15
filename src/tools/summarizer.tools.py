from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

MAX_CHARS = 150000 

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an information compression assistant. The provided text is an output "
        "from a tool ({tool_name}) that is too long for the primary agent's context window.\n\n"
        "Compress the text into a dense, highly informative summary. "
        "Preserve all key data points, facts, metrics, and relevant URLs. "
        "Do not lose structural context."
    ),
    HumanMessage(content="Text to compress:\n\n {content}")
])

summarize_chain = prompt | llm

@tool
def summarize_tool(content: str) -> str:
    """ Summarizes content using cheap llm if content exceeds MAX_CHARS """
    
    if(len(content) > MAX_CHARS):
        try:
            result = summarize_chain.invoke({
                "tool_name": "Web tool",
                "content": content
            })
        except Exception as e:
            return f"Failed to summarize content with error {e}"
    else:
        return content
    
    return result.content + '\n\nCONTENT SUMMARIZED BY MIDDLEWARE'