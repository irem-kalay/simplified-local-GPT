"""
Local Wikipedia RAG Assistant - Streamlit Chat Interface

This module provides a user-friendly chat interface for the Local Wikipedia RAG system.
Users can ask questions about famous people and places, and receive grounded answers
from the local knowledge base.

Features:
- Real-time chat interface with history
- Source chunk display with entity metadata
- Query type classification (person/place/mixed)
- Responsive loading indicators
- Chat history management
"""

import streamlit as st
import warnings
import time
from rag_engine import initialize_rag_engine, prepare_context, generate_answer_stream

# Suppress noisy PyTorch warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Local Wikipedia RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CACHED INITIALIZATION
# ============================================================================

@st.cache_resource
def load_rag_engine():
    """
    Initialize the RAG engine once and cache it.
    
    This function is decorated with @st.cache_resource to ensure that
    the RAG engine (ChromaDB, embedding model, Ollama) is only initialized
    once when the app starts, not on every page reload.
    
    Returns:
        None (initializes globals in rag_engine module)
    """
    try:
        initialize_rag_engine()
    except Exception as e:
        st.error(f"Failed to initialize RAG engine: {str(e)}")
        st.stop()


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """
    Initialize session state for chat history and settings.
    
    Streamlit's session state persists across reruns within the same session,
    allowing us to maintain chat history and user preferences.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "query_types" not in st.session_state:
        st.session_state.query_types = {}


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar():
    """
    Render the sidebar with controls and information.
    """
    with st.sidebar:
        st.markdown("## 🎛️ Controls")

        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_types = {}
            st.success("Chat history cleared!")
            st.rerun()

        st.divider()

        # System information
        st.markdown("## 📊 System Info")
        st.info(
            "**Local Wikipedia RAG Assistant**\n\n"
            "- 🔍 Semantic search powered by embeddings\n"
            "- 🧠 Local LLM Mistral (via Ollama)\n"
            "- 📚 Wikipedia knowledge base\n"
            "- ⚡ Zero external APIs\n"
        )

        st.divider()

        # About section
        st.markdown("## ℹ️ About")
        st.markdown(
            """
            This assistant answers questions about famous **people** and **places**
            using a local knowledge base. All processing happens on your computer.
            
            ### How it works:
            1. Your question is classified (person/place/mixed)
            2. Relevant information is retrieved from the vector database
            3. A local LLM generates an answer grounded in the context
            4. The answer is displayed with source information
            
            ### Example Questions:
            - "Who was Albert Einstein?"
            - "Where is the Eiffel Tower?"
            - "Compare Messi and Ronaldo"
            """
        )

        st.divider()

        # Model information
        st.markdown("## 🤖 Models Used")
        st.markdown(
            """
            - **Embeddings:** all-MiniLM-L6-v2 (384-dim)
            - **LLM:** Mistral (via Ollama)
            - **Vector DB:** Chroma (SQLite backend)
            - **Knowledge Base:** Wikipedia (40 entities)
            """
        )


def render_chat_message(role: str, content: str, query_type: str = None, search_time: float = None, gen_time: float = None):
    """
    Render a chat message with appropriate styling.
    
    Args:
        role: "user" or "assistant"
        content: Message content
        query_type: Optional query classification (person/place/mixed)
        search_time: Time taken for semantic search
        gen_time: Time taken for LLM generation
    """
    with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
        st.markdown(content)

        # Show query type for assistant messages
        if role == "assistant" and query_type:
            query_type_emoji = {
                "person": "👤",
                "place": "📍",
                "mixed": "🔀",
            }
            emoji = query_type_emoji.get(query_type, "❓")
            caption_text = f"{emoji} Query type: {query_type}"
            
            # Süreleri UI'a ekleme kısmı
            if search_time is not None and gen_time is not None:
                if search_time == 0 and gen_time == 0:
                    caption_text += " | ⚡ Cache Hit"
                else:
                    caption_text += f" | ⏱️ Search: {search_time:.2f}s | 🧠 Gen: {gen_time:.2f}s"
                    
            st.caption(caption_text)


def render_source_chunks(sources: list, query_type: str):
    """
    Render retrieved source chunks in an expander.
    
    Args:
        sources: List of source chunk dictionaries
        query_type: Query classification result
    """
    if not sources:
        return

    with st.expander(f"📚 View Retrieved Context ({len(sources)} chunks)", expanded=False):
        # Show query classification
        st.markdown("### Query Classification")
        query_type_description = {
            "person": "👤 About a **person**",
            "place": "📍 About a **place**",
            "mixed": "🔀 **Comparison** or mixed query",
        }
        st.info(query_type_description.get(query_type, "Unknown"))

        # Show source chunks
        st.markdown("### Source Chunks")
        for i, source in enumerate(sources, 1):
            entity_name = source.get("entity_name", "Unknown")
            entity_type = source.get("entity_type", "Unknown")
            chunk_idx = source.get("chunk_index", "0")
            content = source.get("content", "")

            # Truncate content for display
            display_content = content[:200] + "..." if len(content) > 200 else content

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{i}. {entity_name}** ({entity_type.upper()})")
                st.caption(f"Chunk {chunk_idx}")
            with col2:
                st.markdown(f"📄")

            st.markdown(f"> {display_content}\n")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application logic for the Streamlit chat interface.
    """
    # Initialize
    load_rag_engine()
    initialize_session_state()

    # Header
    st.title("🤖 Local Wikipedia RAG Assistant")
    st.markdown(
        """
        Ask me anything about **famous people** and **famous places**.
        I'll search my local knowledge base and provide you with grounded answers.
        """
    )

    # Sidebar
    render_sidebar()

    # Main chat area
    st.divider()

    # Display chat history
    last_assistant_index = max(
        (i for i, message in enumerate(st.session_state.messages) if message["role"] == "assistant"),
        default=-1,
    )

    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]
        query_type = message.get("query_type")
        search_time = message.get("search_time") # YENİ
        gen_time = message.get("gen_time")       # YENİ

        render_chat_message(role, content, query_type, search_time, gen_time)

        # Only show the latest retrieved-context window in the transcript.
      #  if role == "assistant" and i == last_assistant_index:
      #      sources = message.get("sources", [])
      #      render_source_chunks(sources, query_type)

    st.divider()

    # Chat input
    user_input = st.chat_input(
        "Ask me a question about people or places...",
        key="chat_input",
    )

    if user_input:
        # BUG 3 FIX (Doğru Kullanım): Geçmişi, yeni soruyu state'e EKLEMEDEN ÖNCE kopyalıyoruz.
        # Böylece condense_query yeni soruyu sohbet geçmişinin bir parçası sanmaz.
        chat_history = st.session_state.messages.copy()

        # Yeni mesajı state'e ekle
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })

        # Display user message immediately
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        search_time = 0.0
        gen_time = 0.0

        # Generate response with spinner for classify+retrieve
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                from rag_engine import condense_query, _response_cache

                # Sadece son 6 mesajı (3 tur) analiz için yolla
                history_for_condense = chat_history[-6:] if len(chat_history) > 0 else []

                # ADIM 1: Önce soruyu bağlamlı hale getir (Standalone Query)
                # SADECE soruyu çözen fonksiyonu çağırıyoruz (tüm sistemi değil)
                standalone_query = condense_query(user_input, history_for_condense) if history_for_condense else user_input
                
                # LLM'in fazladan ekleyebileceği prefixleri temizle (Güvenlik için eklendi)
                if standalone_query:
                    standalone_query = standalone_query.replace("Standalone question:", "").replace("Standalone question", "").replace('"', '').strip()

                # ADIM 2: CACHE KONTROLÜ (Hem ham soru hem de bağlamlı soru üzerinden - ÇİFT KONTROL)
                raw_cache_key = user_input.strip().lower()
                standalone_cache_key = (standalone_query or user_input).strip().lower()
                
                # Hangi anahtarda bulursak onu kullanacağız
                matched_cache_key = None
                if raw_cache_key in _response_cache:
                    matched_cache_key = raw_cache_key
                elif standalone_cache_key in _response_cache:
                    matched_cache_key = standalone_cache_key
                
                if matched_cache_key:
                    cached = _response_cache[matched_cache_key]
                    answer = cached.get("answer", "I don't know.")
                    query_type = cached.get("query_type", "unknown")
                    sources = cached.get("sources", [])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "query_type": query_type,
                        "sources": sources,
                        "search_time": 0.0, 
                        "gen_time": 0.0,    
                    })
                    render_chat_message("assistant", answer, query_type, 0.0, 0.0)

                    render_source_chunks(sources, query_type)
                    
                    # Eğer cache'ten gelirse işlemi burada bitir.
                    st.stop()
                else:
                    t_start_search = time.time()
                    # ADIM 3: Eğer cache'te yoksa, tüm hazırlığı yap.
                    # BUG 2 FIX: Artık user_input yerine standalone_query gönderiyoruz
                    # ve history olarak da aynı history_for_condense kullanıyoruz
                    _, query_type, context_text, sources, history_text, error = prepare_context(
                        standalone_query, history_for_condense
                    )

                    search_time = time.time() - t_start_search

                    if error:
                        answer = f"⚠️ Error: {error}"
                        st.warning(answer)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "query_type": "error",
                            "sources": [],
                        })
                        st.stop()

            except Exception as e:
                error_message = f"❌ Error processing your question: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "query_type": "error",
                    "sources": [],
                })
                error = error_message
                query_type = "error"
                sources = []
                context_text = ""
                history_text = ""

        # Stream the LLM answer outside the spinner
        if not locals().get("error"):
            try:
                query_type_emoji = {"person": "👤", "place": "📍", "mixed": "🔀"}
                with st.chat_message("assistant", avatar="🤖"):
                    placeholder = st.empty()
                    full_answer = ""

                    t_start_gen = time.time()
                    
                    # user_input yerine standalone_query akışa veriliyor
                    # İsteğin üzerine history_text korundu, silinmedi.
                    for token in generate_answer_stream(
                        standalone_query, context_text, query_type, history_text
                    ):
                        full_answer += token
                        placeholder.markdown(full_answer + "▌")

                    gen_time = time.time() - t_start_gen
                    
                    placeholder.markdown(full_answer)
                    answer = full_answer if full_answer else "I don't know"
                    if query_type:
                        emoji = query_type_emoji.get(query_type, "❓")
                        st.caption(f"{emoji} Query type: {query_type} | ⏱️ Search: {search_time:.2f}s | 🧠 Gen: {gen_time:.2f}s")

                        
                # Save to session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "query_type": query_type,
                    "sources": sources,
                    "search_time": search_time,
                    "gen_time": gen_time,
                })
                
                # YENİ CACHE KAYDI: Hem ham soru hem de bağlamlı soru ile çift kayıt yapılıyor.
                cache_data_to_store = {
                    "query": standalone_query,
                    "answer": answer,
                    "query_type": query_type,
                    "sources": sources,
                    "context": context_text,  # BUG 6 FIX: Artık boş değil, gerçek context kaydediliyor
                    "error": None,
                }
                _response_cache[raw_cache_key] = cache_data_to_store
                _response_cache[standalone_cache_key] = cache_data_to_store

                render_source_chunks(sources, query_type)

            except Exception as e:
                error_message = f"❌ Error generating answer: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "query_type": "error",
                    "sources": [],
                })
# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()