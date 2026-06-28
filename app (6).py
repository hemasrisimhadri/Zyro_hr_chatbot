import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

# ── Constants ───────────────────────────────────────────────
REFUSAL_MESSAGE = (
    "I'm sorry, I can only answer HR-related questions based on "
    "Zyro Dynamics' internal policy documents. Your question appears "
    "to be outside the scope of HR policies. Please ask me about "
    "topics like leave, salary, work from home, performance reviews, "
    "code of conduct, or any other HR-related policies."
)

OUT_OF_SCOPE_KEYWORDS = [
    "weather", "cricket", "football", "movie", "recipe",
    "stock", "crypto", "bitcoin", "politics", "news",
    "sports", "music", "game", "restaurant",
    "shopping", "fashion", "celebrity", "gossip",
    "coding", "programming", "javascript",
    "relationship", "marriage", "dating",
    "diet", "workout", "astrology", "horoscope"
]

HR_KEYWORDS = [
    "leave", "salary", "policy", "employee", "hr",
    "work from home", "wfh", "hybrid", "remote",
    "performance", "review", "appraisal", "bonus",
    "incentive", "reimbursement", "travel", "expense",
    "onboarding", "separation", "resignation", "termination",
    "probation", "notice period", "posh", "harassment",
    "conduct", "ethics", "data security", "compensation",
    "benefits", "maternity", "paternity", "sick leave",
    "earned leave", "casual leave", "holiday", "payroll",
    "ctc", "grade", "promotion", "pip", "increment",
    "joining", "full and final", "pf", "gratuity", "insurance"
]

# ── Load and Build RAG Pipeline ─────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading HR policy documents...")
def build_rag_pipeline():
    # Load PDFs
    loader = PyPDFDirectoryLoader("docs/")
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Build embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6},
    )

    # Build LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=512,
        api_key=os.environ.get("GROQ_API_KEY")
    )

    # Build prompt
    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", \"\"\"You are an HR assistant for Zyro Dynamics Pvt. Ltd.
Your ONLY knowledge source is the HR policy document excerpts provided below.
STRICT RULES:
1. Answer ONLY using the information in the context below.
2. Do NOT use any outside knowledge or assumptions.
3. Always mention which policy document your answer comes from.
4. Keep answers clear, professional, and concise.
5. If the context does not contain enough information to answer, say:
   'I could not find specific information about this in the HR policy documents.'
--- HR POLICY CONTEXT ---
{context}
--- END CONTEXT ---\"\"\"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        sections = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            source = source.split("/")[-1]
            sections.append(f"[Excerpt {i} — {source}]\n{doc.page_content}")
        return "\n\n".join(sections)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever

# ── Guardrails ───────────────────────────────────────────────
def is_hr_related(question):
    question_lower = question.lower().strip()
    for keyword in HR_KEYWORDS:
        if keyword in question_lower:
            return True
    for keyword in OUT_OF_SCOPE_KEYWORDS:
        if keyword in question_lower:
            return False
    return True

def answer_with_guardrails(question, chain, retriever):
    if not is_hr_related(question):
        return REFUSAL_MESSAGE, []
    try:
        response = chain.invoke(question)
        docs = retriever.invoke(question)
        return response, docs
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}", []

# ── UI ───────────────────────────────────────────────────────
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about Zyro Dynamics HR policies!")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/human-resources.png", width=80)
    st.markdown("### 📋 Topics I Can Help With")
    st.markdown(
        \"\"\"
- 🏖️ Leave Policy
- 🏠 Work From Home Policy
- 💰 Salary & Compensation
- 📊 Performance Reviews
- 🤝 Code of Conduct
- 🔒 IT & Data Security
- 🚨 POSH Policy
- ✈️ Travel & Expenses
- 🎯 Onboarding & Separation
        \"\"\"
    )
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Build pipeline
chain, retriever = build_rag_pipeline()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I'm your Zyro Dynamics HR Assistant. How can I help you today?"
    })

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 View Sources"):
                for src in msg["sources"]:
                    st.write(f"• {src}")

# Chat input
if question := st.chat_input("Ask your HR question here..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching HR policies..."):
            answer, docs = answer_with_guardrails(question, chain, retriever)
            sources = list(set([
                d.metadata.get("source", "Unknown").split("/")[-1]
                for d in docs
            ])) if docs else []

        st.write(answer)

        if sources:
            with st.expander("📄 View Sources"):
                for src in sources:
                    st.write(f"• {src}")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })