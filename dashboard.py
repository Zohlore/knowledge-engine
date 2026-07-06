# Streamlit UI
import streamlit as st
import pandas as pd
import time
from pathlib import Path
from orchestrator import orchestrator
from ingest import ingestor
from vector_store import vector_store
from logger import logger

st.set_page_config(
    page_title="Knowledge Engine",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Multi-Agent Knowledge Engine")
st.markdown("""
Query your documents with AI-powered retrieval and verification.
*Citations and grounding verification included.*
""")

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    
    stats = vector_store.get_stats()
    st.metric("Documents Indexed", stats.get('points_count', 0))
    st.metric("Status", stats.get('status', 'unknown'))
    
    st.markdown("---")
    st.header("📁 Document Management")
    
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file:
        if st.button("📥 Ingest Document"):
            with st.spinner("Ingesting document..."):
                # Save file
                save_path = Path("data/documents") / uploaded_file.name
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest
                result = ingestor.ingest_pdf(str(save_path))
                if result.success:
                    st.success(f"✅ Ingested: {uploaded_file.name} ({result.chunks_created} chunks)")
                else:
                    st.error(f"❌ Failed: {result.error}")
    
    st.markdown("---")
    st.caption("Built with LangChain, Qdrant, OpenAI")

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Query", "📊 Analytics", "📁 Documents"])

with tab1:
    st.subheader("Ask a Question")
    
    query = st.text_area("Enter your question:", height=100, 
                         placeholder="e.g., What is the process for handling customer complaints?")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_btn and query:
        with st.spinner("Processing..."):
            try:
                start_time = time.time()
                response = orchestrator.query(query)
                elapsed = (time.time() - start_time) * 1000
                
                # Display results
                st.subheader("💡 Answer")
                st.write(response.answer)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Confidence", f"{response.confidence_score*100:.1f}%")
                col2.metric("Grounded", "✅" if response.grounded else "⚠️")
                col3.metric("Sources", len(response.citations))
                col4.metric("Latency", f"{elapsed:.0f}ms")
                
                # Verification notes
                if response.verification_notes:
                    with st.expander("📋 Verification Notes"):
                        for note in response.verification_notes:
                            st.write(f"- {note}")
                
                # Citations - UPDATED: use attribute access, not .get()
                if response.citations:
                    st.subheader("📚 Sources")
                    for i, citation in enumerate(response.citations, 1):
                        with st.expander(f"Source {i} (Similarity: {citation.similarity:.2f})"):
                            st.write(citation.text)
                            st.caption(f"Source: {citation.source}")
                
                # Raw chunks
                if response.source_chunks:
                    with st.expander("🔍 View Retrieved Chunks"):
                        for i, chunk in enumerate(response.source_chunks[:5], 1):
                            st.write(f"**Chunk {i}** (Score: {chunk.similarity_score:.3f})")
                            st.text(chunk.chunk.content[:300] + "...")
                            st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                logger.error(f"Query error: {e}")

with tab2:
    st.subheader("📊 Analytics")
    
    stats = vector_store.get_stats()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Documents", stats.get('points_count', 0))
    col2.metric("Vector Size", stats.get('vectors_count', 0))
    col3.metric("Status", stats.get('status', 'unknown'))
    
    st.markdown("### 💾 Cache Stats")
    # Add cache stats here if implemented

with tab3:
    st.subheader("📁 Documents")
    
    # List documents
    doc_dir = Path("data/documents")
    if doc_dir.exists():
        docs = list(doc_dir.glob("*.pdf"))
        if docs:
            df = pd.DataFrame({
                'Document': [d.name for d in docs],
                'Size (KB)': [d.stat().st_size / 1024 for d in docs],
                'Modified': [pd.to_datetime(d.stat().st_mtime, unit='s') for d in docs]
            })
            st.dataframe(df)
        else:
            st.info("No documents found. Upload a PDF in the sidebar.")
    else:
        st.info("Documents directory not found. Upload a PDF to create it.")