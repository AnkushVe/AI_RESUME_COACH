import streamlit as st
import os
import pdfplumber
from docx import Document
from rake_nltk import Rake
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai


openai.api_key = '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'
job_descriptions_df = pd.read_csv('job_descriptions_with_keywords.csv')

nltk.download('punkt')


def parse_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text


def parse_docx(file_path):
    doc = Document(file_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text
    return text


def parse_txt(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text


def parse_resume(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return parse_pdf(file_path)
    elif file_extension.lower() == '.docx':
        return parse_docx(file_path)
    elif file_extension.lower() == '.txt':
        return parse_txt(file_path)
    else:
        raise ValueError("Unsupported file format")


def extract_keywords(text):
    r = Rake()  # Initialize RAKE
    r.extract_keywords_from_text(text)

    # Get ranked phrases with scores
    ranked_phrases = r.get_ranked_phrases_with_scores()
    return ranked_phrases


def calculate_matching_score(resume_keywords, job_keywords):
    # Convert keywords to TF-IDF vectors
    vectorizer = TfidfVectorizer(lowercase=True)
    vectors = vectorizer.fit_transform(
        [str(resume_keywords), str(job_keywords)])

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(vectors)[0, 1]

    scaling_factor = 0.35
    offset = 60

    matching_score = (cosine_similarities * 100 * scaling_factor)+offset
    matching_score = round(matching_score, 2)
    return matching_score


def main():
    st.title("AI Resume Coach :robot:")
    st.caption("Assisting You in Building Your Resume")


    st.header("About Me")
    st.text('I am an AI-powered Resume Coach here to assist you in creating a strong resume.')
    st.text('I will provide guidance on resume content and formatting.')

    st.subheader("Let's Get Started!")
    st.text("Please upload your resume below:")
    uploaded_file = st.file_uploader(
        "Upload a resume file", type=['pdf', 'docx', 'txt'])

    if uploaded_file:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        parsed_resume = parse_resume(file_path)
        resume_keywords = extract_keywords(parsed_resume)

        matching_scores = []
        for _, job_row in job_descriptions_df.iterrows():
            job_category = job_row['category']
            job_keywords = job_row[['category', 'description']].values.astype(
                str).tolist()
            job_keywords = " ".join(job_keywords)
            matching_score = calculate_matching_score(
                resume_keywords, job_keywords)
            matching_scores.append((job_category, matching_score))

        matching_scores.sort(key=lambda x: x[1], reverse=True)

        top_category, top_score = matching_scores[0]
        prompt = f"Generate a detailed feedback report for a resume applying to the {top_category} role with a matching score of {top_score:.2f}%. Provide detailed feedback in a paragraph format covering various aspects such as action verbs, skills, projects, experiences, quantification of results, language use, conciseness, customization for the role, and proofreading."

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=4000
        )

        feedback_text = response['choices'][0]['text']
        feedback_lines = feedback_text.split('. ')

        st.write(f"Matching Score: {top_score}")
        st.write("Feedback:")
        for line in feedback_lines:
            st.write(line)


if __name__ == '__main__':
    main()
