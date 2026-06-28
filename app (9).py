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

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

REFUSAL_MESSAGE = (
    "I am sorry, I can only answer HR-related questions based on "
    "Zyro Dynamics internal policy documents. Your question appears "
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

@st.cache_resource(show_spinner="Loading HR policy documents...")
def build_rag_pipeline():

    # ── Check docs folder exists ────────────────────────────
    docs_path = "docs/"
    if not os.path.exists(docs_path):
        st.error("docs/ folder not found! Please add HR PDFs to the docs/ folder in your repo.")
        st.stop()

    pdf_files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    if len(pdf_files) == 0:
        st.error("No PDF files found in docs/ folder! Please upload the 11 HR policy PDFs.")
        st.stop()

    st.info(f"Found {len(pdf_files)} PDF files. Loading...")

    # ── Load PDFs ───────────────────────────────────────────
    loader = PyPDFDirectoryLoader(docs_path)
    documents = loader.load()

    if len(documents) == 0:
        st.error("PDFs loaded but no content extracted! Check if PDFs are readable.")
        st.stop()

    # ── Chunk documents ─────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # ── Build embeddings ────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Build FAISS vector store ────────────────────────────
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6},
    )

    # ── Initialize LLM ──────────────────────────────────────
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        st.error("GROQ_API_KEY not found! Add it in Streamlit Cloud Secrets.")
        st.stop()

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=512,
        api_key=groq_key
    )

    # ── Build prompt ────────────────────────────────────────
    system_msg = (
        "You are an HR assistant for Zyro Dynamics Pvt. Ltd. "
        "Your ONLY knowledge source is the HR policy document excerpts provided below. "
        "STRICT RULES: "
        "1. Answer ONLY using the information in the context below. "
        "2. Do NOT use any outside knowledge or assumptions. "
        "3. Always mention which policy document your answer comes from. "
        "4. Keep answers clear, professional, and concise. "
        "5. If the context does not contain enough information, say: "
        "I could not find specific information about this in the HR policy documents. "
        "--- HR POLICY CONTEXT --- "
        "{context} "
        "--- END CONTEXT ---"
    )

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        sections = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            source = source.split("/")[-1]
            sections.append(f"[Excerpt {i} - {source}]\n{doc.page_content}")
        return "\n\n".join(sections)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever

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

st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about Zyro Dynamics HR policies!")

with st.sidebar:
    st.markdown("### 📋 Topics I Can Help With")
    st.markdown(
        "- 🏖️ Leave Policy\n"
        "- 🏠 Work From Home\n"
        "- 💰 Salary and Compensation\n"
        "- 📊 Performance Reviews\n"
        "- 🤝 Code of Conduct\n"
        "- 🔒 IT and Data Security\n"
        "- 🚨 POSH Policy\n"
        "- ✈️ Travel and Expenses\n"
        "- 🎯 Onboarding and Separation\n"
    )
    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

chain, retriever = build_rag_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I am your Zyro Dynamics HR Assistant. How can I help you today?"
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Sources"):
                for src in msg["sources"]:
                    st.write(f"- {src}")

if question := st.chat_input("Ask your HR question here..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            answer, docs = answer_with_guardrails(question, chain, retriever)
            sources = list(set([
                d.metadata.get("source", "Unknown").split("/")[-1]
                for d in docs
            ])) if docs else []

        st.write(answer)
        if sources:
            with st.expander("View Sources"):
                for src in sources:
                    st.write(f"- {src}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
