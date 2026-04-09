import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()
# Title
st.title("📄 Resume Screening System")

# Upload dataset
uploaded_file = st.file_uploader("Upload Resume Dataset (CSV)", type=["csv"])

# Job description input
job_desc = st.text_area("Enter Job Description")

# Clean function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# Skill list
skills_list = [
    'python','java','sql','machine learning','data analysis',
    'nlp','deep learning','html','css','javascript',
    'excel','power bi','tensorflow','pandas','numpy'
]

def extract_skills(text):
    return [skill for skill in skills_list if skill in text]

# Run button
if st.button("Analyze Resumes"):
    if uploaded_file is not None and job_desc != "":
        
        df = pd.read_csv(uploaded_file)

        # Use Resume_str column
        df['cleaned_resume'] = df['Resume_str'].apply(clean_text)
        df['skills'] = df['cleaned_resume'].apply(extract_skills)

        job_clean = clean_text(job_desc)

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(
            df['cleaned_resume'].tolist() + [job_clean]
        )

        cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        df['score'] = cosine_sim[0]

        df_sorted = df.sort_values(by='score', ascending=False)

        st.subheader("🏆 Top Candidates")
        st.write(df_sorted[['ID','Category','score']].head(10))

        # Skill gap
        def skill_gap(candidate_skills):
            job_skills = extract_skills(job_clean)
            return list(set(job_skills) - set(candidate_skills))

        df['missing_skills'] = df['skills'].apply(skill_gap)

        st.subheader("⚠️ Skill Gap")
        st.write(df[['skills','missing_skills']].head())

    else:
        st.warning("Please upload dataset and enter job description")
