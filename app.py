import streamlit as st

from rag import (
    CHAT_MODEL,
    DB_PATH,
    EMBEDDING_MODEL,
    delete_document,
    generate_answer,
    index_document,
    list_documents,
    retrieve,
)


st.set_page_config(page_title="Family Health Archive", page_icon="+", layout="wide")
st.title("Family Health Archive")
st.caption("Private, local search across family medical records")
st.warning(
    "Information retrieval only, not medical advice. Check answers against the original record and consult a clinician.",
    icon="⚕️",
)

documents = list_documents(DB_PATH)
members = sorted({document["member"] for document in documents}, key=str.casefold)

with st.sidebar:
    st.header("Add records")
    member_name = st.text_input("Family member", placeholder="e.g. Alex Smith")
    uploads = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Index selected files", type="primary", disabled=not uploads):
        if not member_name.strip():
            st.error("Enter the family member's name first.")
        else:
            with st.spinner("Extracting text and creating local embeddings..."):
                indexed = 0
                for upload in uploads:
                    try:
                        count = index_document(DB_PATH, member_name, upload.name, upload.getvalue())
                        indexed += 1
                        st.success(f"Indexed {upload.name} ({count} text chunks)")
                    except Exception as error:
                        st.error(f"Could not index {upload.name}: {error}")
            if indexed:
                st.rerun()

    st.divider()
    st.header("Indexed sources")
    if documents:
        selected_source = st.selectbox(
            "Choose a source to remove",
            documents,
            format_func=lambda source: f"{source['member']} · {source['filename']}",
        )
        if st.button("Remove selected source"):
            delete_document(DB_PATH, selected_source["source_hash"])
            st.rerun()
    else:
        st.caption("No records indexed yet.")

    st.divider()
    st.caption(f"Local models: {CHAT_MODEL} · {EMBEDDING_MODEL}")

if not members:
    st.info("Add a text-based PDF or a TXT/Markdown note to begin.")
    st.stop()

scope = st.selectbox("Search records for", ["All family members", *members])
with st.form("question_form"):
    question = st.text_area("Question", placeholder="Which medications are mentioned for Alex?")
    submitted = st.form_submit_button("Search records", type="primary")

if submitted and not question.strip():
    st.warning("Enter a question before searching.")

if submitted and question.strip():
    person_filter = None if scope == "All family members" else scope
    try:
        with st.spinner("Searching local records and preparing an answer..."):
            sources = retrieve(DB_PATH, question.strip(), person_filter)
            if not sources:
                st.info("No indexed record text was found for this search.")
            else:
                answer = generate_answer(question.strip(), sources)
                st.subheader("Answer")
                st.write(answer)
                with st.expander("Retrieved source excerpts", expanded=True):
                    for index, source in enumerate(sources, start=1):
                        st.markdown(f"**Source {index}: {source['filename']} · page {source['page']}**")
                        st.write(source["content"])
    except Exception as error:
        st.error(str(error))