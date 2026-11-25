import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import io
import json
import os
import html

# -------------------------------
# App styling
# -------------------------------
st.set_page_config(page_title="AI Comment Analyzer", page_icon="🤖", layout="wide")
st.markdown("""
<style>
    .stApp { background: radial-gradient(ellipse at 70% 0%, #13202a 0%, #0c1117 48%, #0a0e13 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #111a22 0%, #0c1218 100%); border-right: 1px solid #26343c; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #e8f0f4; }
    .block-container { padding-top: .8rem; max-width: 1120px; }
    .block-container h1 { margin-top: .25rem; }
    h1, h2, h3 { letter-spacing: -0.025em; }
    div[data-testid="stMetric"] { background: #121c24; border: 1px solid #26343c; padding: 1rem; border-radius: 14px; }
    div[data-testid="stMetricLabel"] { color: #a9bac3; }
    div.stButton > button { border-radius: 9px; }
    [data-testid="stTabs"] button { font-weight: 600; }
    [data-testid="stImage"] img { border-radius: 14px; }
    div[data-testid="stDownloadButton"] button { width: 100%; }
    @media (max-width: 768px) {
        .block-container { padding: .9rem 1.25rem 2rem; }
        [data-testid="stSidebar"] img { max-width: 230px; margin: 0 auto; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .15rem; }
        [data-testid="stTabs"] button { padding: .55rem .7rem; }
    }
    @media (max-width: 480px) {
        .block-container { padding: .75rem .8rem 1.5rem; }
        h1 { font-size: 1.8rem !important; line-height: 1.15 !important; }
        h2 { font-size: 1.45rem !important; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { overflow-x: auto; }
        [data-testid="stTabs"] button { white-space: nowrap; }
        div[data-testid="stMetric"] { padding: .8rem; }
        div[data-testid="stMetricValue"] { font-size: 1.5rem; }
        div.stButton > button, div[data-testid="stDownloadButton"] button { min-height: 44px; }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Helper: Save & Load Comments
# -------------------------------
COMMENTS_FILE = "comments_data.json"

def save_comments():
    with open(COMMENTS_FILE, "w") as f:
        json.dump(st.session_state['all_comments'], f)

def load_comments():
    if os.path.exists(COMMENTS_FILE):
        with open(COMMENTS_FILE, "r") as f:
            st.session_state['all_comments'] = json.load(f)
    else:
        st.session_state['all_comments'] = []

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.image("assets/ai-dashboard.png", use_container_width=True)
st.sidebar.title("AI Dashboard")
page = st.sidebar.radio("Navigate", ["Home", "Comments Analyzer", "Analytics", "About"])

# -------------------------------
# Load Sentiment Model
# -------------------------------
@st.cache_resource(show_spinner="Loading the sentiment model…")
def load_classifier():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_classifier()

# Initialize session state
if 'all_comments' not in st.session_state:
    load_comments()

# -------------------------------
# HOME PAGE
# -------------------------------
if page == "Home":
    st.markdown("<h1 style='text-align:center;color:#eaf3f7;margin-bottom:.25rem'>Understand the conversation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#a9bac3;font-size:1.08rem;margin-bottom:1.6rem'>AI-powered sentiment analysis for comments, made simple.</p>", unsafe_allow_html=True)
    _, banner_col, _ = st.columns([2, 5, 2])
    with banner_col:
        st.image("assets/ai-dashboard.png", use_container_width=True)
    st.markdown("### Your comment insights, at a glance")
    st.write("Add comments to classify sentiment, explore patterns, and export a clear report.")
    total = len(st.session_state['all_comments'])
    positive = sum(c['classification'] == 'POSITIVE' for c in st.session_state['all_comments'])
    negative = sum(c['classification'] == 'NEGATIVE' for c in st.session_state['all_comments'])
    m1, m2, m3 = st.columns(3)
    m1.metric("Comments analyzed", total)
    m2.metric("Positive", positive)
    m3.metric("Negative", negative)
    st.info("Choose **Comments Analyzer** in the sidebar to add your first comment.")

# -------------------------------
# COMMENTS ANALYZER PAGE WITH TABS
# -------------------------------
elif page == "Comments Analyzer":
    st.title("Comments Analyzer")
    tab1, tab2 = st.tabs(["➕ Add Comment", "📋 View Comments"])

    # --- Tab 1: Add Comment ---
    with tab1:
        if 'comment_input' not in st.session_state:
            st.session_state['comment_input'] = "Amazing Video"

        comment = st.text_input("Enter your comment", value=st.session_state['comment_input'], key="comment_box")

        if st.button("Add Comment"):
            result = classifier(comment)[0]['label']
            comment_dic = {'comment': comment, 'classification': result}
            st.session_state['all_comments'].append(comment_dic)
            save_comments()
            st.success("Comment added successfully!")

            st.session_state['comment_input'] = ""
            st.rerun()

    # --- Tab 2: View Comments ---
    with tab2:
        sort_option = st.selectbox("Sort Comments by Sentiment:", ["None", "Positive", "Negative"])
        search_term = st.text_input("🔍 Search comments")

        def display_comments(comments):
            for idx, comment in enumerate(comments):
                if comment['classification'] == 'POSITIVE':
                    accent_color = '#35c98b'
                elif comment['classification'] == 'NEGATIVE':
                    accent_color = '#ff7185'
                else:
                    accent_color = '#a9bac3'

                comment_html = f"""
                <div style="max-width:820px;margin:8px 0;padding:12px 15px;border-radius:12px;
                border:1px solid #26343c;border-left:4px solid {accent_color};background:#121c24">
                    <div style="font-weight:700;color:{accent_color};font-size:11px;letter-spacing:.06em;margin-bottom:5px">
                        {html.escape(comment['classification'])}</div>
                    <div style="font-size:14px;color:#e8f0f4;line-height:1.45;overflow-wrap:anywhere">
                        {html.escape(comment['comment'])}</div>
                </div>
                """
                st.write(comment_html, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✏️ Edit", key=f"edit_{idx}", use_container_width=True):
                        st.session_state['edit_index'] = idx
                        st.session_state['edit_text'] = comment['comment']
                        st.rerun()
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{idx}", use_container_width=True):
                        st.session_state['all_comments'].pop(idx)
                        save_comments()
                        st.rerun()

        # Handle editing
        if 'edit_index' in st.session_state:
            edit_idx = st.session_state['edit_index']
            st.info(f"Editing comment #{edit_idx + 1}")
            new_text = st.text_input("Edit your comment:", value=st.session_state['edit_text'])
            if st.button("💾 Save Changes"):
                result = classifier(new_text)[0]['label']
                st.session_state['all_comments'][edit_idx] = {'comment': new_text, 'classification': result}
                save_comments()
                st.success("Comment updated successfully!")
                del st.session_state['edit_index']
                del st.session_state['edit_text']
                st.rerun()
            if st.button("❌ Cancel Edit"):
                del st.session_state['edit_index']
                del st.session_state['edit_text']
                st.rerun()

        filtered_comments = st.session_state['all_comments']
        if sort_option == "Positive":
            filtered_comments = [c for c in filtered_comments if c['classification'] == 'POSITIVE']
        elif sort_option == "Negative":
            filtered_comments = [c for c in filtered_comments if c['classification'] == 'NEGATIVE']

        if search_term:
            filtered_comments = [c for c in filtered_comments if search_term.lower() in c['comment'].lower()]

        if filtered_comments:
            display_comments(filtered_comments)
        else:
            st.info("No comments yet. Add some in the 'Add Comment' tab!")

# -------------------------------
# ANALYTICS PAGE WITH EXPORT + PDF
# -------------------------------
elif page == "Analytics":
    st.title("Sentiment Analytics")
    if st.session_state['all_comments']:
        df = pd.DataFrame(st.session_state['all_comments'])
        positive_count = int((df['classification'] == 'POSITIVE').sum())
        negative_count = int((df['classification'] == 'NEGATIVE').sum())
        metric_cols = st.columns(3)
        metric_cols[0].metric("Total comments", len(df))
        metric_cols[1].metric("Positive", positive_count)
        metric_cols[2].metric("Negative", negative_count)

        sentiment_counts = df['classification'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        fig_bar = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment',
                         title="Sentiment Distribution", template="plotly_dark",
                         color_discrete_map={'POSITIVE':'#35c98b','NEGATIVE':'#ff7185'})
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=12, r=12, t=55, b=12), legend_title_text="")
        fig_pie = px.pie(sentiment_counts, names='Sentiment', values='Count',
                         title="Sentiment Breakdown", color='Sentiment',
                         color_discrete_map={'POSITIVE':'#35c98b','NEGATIVE':'#ff7185'})
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=12, r=12, t=55, b=12),
                              legend_title_text="")
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.plotly_chart(fig_bar, use_container_width=True)
        with chart_cols[1]:
            st.plotly_chart(fig_pie, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="AI Sentiment Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Total Comments: {len(df)}", ln=True)
        pdf.cell(200, 10, txt=f"Positive: {(df['classification']=='POSITIVE').sum()}", ln=True)
        pdf.cell(200, 10, txt=f"Negative: {(df['classification']=='NEGATIVE').sum()}", ln=True)
        pdf.cell(200, 10, txt="----------------------------------------", ln=True)
        pdf.cell(200, 10, txt="Comments Summary:", ln=True)
        for i, row in df.iterrows():
            pdf.multi_cell(0, 10, f"{row['classification']}: {row['comment']}")
        pdf_output = io.BytesIO(pdf.output(dest='S').encode('latin-1'))
        export_cols = st.columns(2)
        with export_cols[0]:
            st.download_button("📥 Download CSV", csv, "comments_sentiment.csv", "text/csv", use_container_width=True)
        with export_cols[1]:
            st.download_button("📄 Download PDF Report", pdf_output, "sentiment_report.pdf", "application/pdf", use_container_width=True)

    else:
        st.warning("No comments added yet. Go to 'Comments Analyzer' to add some!")

# -------------------------------
# ABOUT PAGE
# -------------------------------
elif page == "About":
    st.title("About This Dashboard")
    st.markdown("""
    **Developer:** Sneha  
    **Purpose:** project demonstrating AI‑powered sentiment analysis  
    **Tools Used:** Streamlit, HuggingFace Transformers, Plotly, FPDF  
    """)
    st.markdown("<hr style='border:1px solid #4CAF50;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#4CAF50;'>Thank you for visiting the AI Dashboard!</p>", unsafe_allow_html=True)

