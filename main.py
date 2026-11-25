import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import io
import json
import os

# -------------------------------
# App styling
# -------------------------------
st.set_page_config(page_title="AI Comment Analyzer", page_icon="🤖", layout="wide")
st.markdown("""
<style>
    .stApp { background: radial-gradient(ellipse at 70% 0%, #13202a 0%, #0c1117 48%, #0a0e13 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #111a22 0%, #0c1218 100%); border-right: 1px solid #26343c; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #e8f0f4; }
    .block-container { padding-top: 2.2rem; max-width: 1120px; }
    h1, h2, h3 { letter-spacing: -0.025em; }
    div[data-testid="stMetric"] { background: #121c24; border: 1px solid #26343c; padding: 1rem; border-radius: 14px; }
    div[data-testid="stMetricLabel"] { color: #a9bac3; }
    div.stButton > button { border-radius: 9px; }
    [data-testid="stTabs"] button { font-weight: 600; }
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
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# Initialize session state
if 'all_comments' not in st.session_state:
    load_comments()

# -------------------------------
# HOME PAGE
# -------------------------------
if page == "Home":
    st.markdown("<h1 style='text-align:center;color:#eaf3f7;margin-bottom:.25rem'>Understand the conversation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#a9bac3;font-size:1.08rem;margin-bottom:1.6rem'>AI-powered sentiment analysis for comments, made simple.</p>", unsafe_allow_html=True)
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
                # Darker shades for better contrast
                if comment['classification'] == 'POSITIVE':
                    bg_color = '#7FB77E'  # darker green
                elif comment['classification'] == 'NEGATIVE':
                    bg_color = '#D98880'  # darker red
                else:
                    bg_color = '#CFCFCF'  # neutral gray

                comment_html = f"""
                <div style="margin:10px;padding:15px;border-radius:10px;
                box-shadow:0px 4px 10px rgba(0,0,0,0.2);
                background-color:{bg_color};padding-left:25px;padding-top:15px;padding-bottom:15px">
                    <div style="font-weight:bold;color:#222;font-size:16px;">👤 Username</div>
                    <div style="font-size:15px;color:#000;">{comment['comment']}</div>
                </div>
                """
                st.write(comment_html, unsafe_allow_html=True)

                col1, col2 = st.columns([0.15, 0.15])
                with col1:
                    if st.button("✏️ Edit", key=f"edit_{idx}"):
                        st.session_state['edit_index'] = idx
                        st.session_state['edit_text'] = comment['comment']
                        st.rerun()
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{idx}"):
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
        sentiment_counts = df['classification'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        fig_bar = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment',
                         title="Sentiment Distribution", template="plotly_dark")
        st.plotly_chart(fig_bar)

        fig_pie = px.pie(sentiment_counts, names='Sentiment', values='Count',
                         title="Sentiment Breakdown", color='Sentiment',
                         color_discrete_map={'POSITIVE':'#4CAF50','NEGATIVE':'#F44336'})
        st.plotly_chart(fig_pie)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Comments Data (CSV)", csv, "comments_sentiment.csv", "text/csv")

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
        st.download_button("📄 Download PDF Report", pdf_output, "sentiment_report.pdf", "application/pdf")

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

