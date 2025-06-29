from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.credentials import Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
import PyPDF2
import os
import re
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for Flask/web apps
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64
from dotenv import load_dotenv

load_dotenv()

# ---------------- IBM CONFIG ----------------
API_KEY = os.getenv("IBM_API_KEY")
PROJECT_ID = os.getenv("IBM_PROJECT_ID")
URL = os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com")
MODEL_ID = "ibm/granite-3-8b-instruct"

# ---------------- PDF EXTRACT ----------------
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text.strip()

# ---------------- GRANITE QUERY ----------------
def query_granite(prompt):
    creds = Credentials(api_key=API_KEY, url=URL)
    model = ModelInference(
        model_id=MODEL_ID,
        credentials=creds,
        project_id=PROJECT_ID
    )
    try:
        result = model.generate_text(
            prompt=prompt,
            params={
                GenParams.MAX_NEW_TOKENS: 300,
                GenParams.TEMPERATURE: 0.7
            }
        )
        if isinstance(result, str):
            return result
        elif isinstance(result, dict) and "results" in result and result["results"]:
            return result["results"][0]["generated_text"]
        else:
            return "Error: Unexpected response format."
    except Exception as e:
        print("Error in query_granite:", e)
        return "Error: Could not generate response."

# ---------------- VISUALIZATION TOOLS ----------------
def create_risk_gauge(score):
    """Create a risk score gauge visualization"""
    plt.figure(figsize=(6, 3))
    ax = plt.subplot(111)
    
    # Create color zones
    ax.barh(0, 3, height=0.5, color='green', alpha=0.3)
    ax.barh(0, 4, left=3, height=0.5, color='yellow', alpha=0.3)
    ax.barh(0, 3, left=7, height=0.5, color='red', alpha=0.3)
    
    # Add needle
    ax.arrow(score, 0, 0, 0.4, head_width=0.3, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # Set properties
    plt.xlim(0, 10)
    plt.ylim(-0.5, 1)
    plt.xticks(np.arange(0, 11, 1))
    plt.title(f"Risk Score: {score}/10", fontweight='bold')
    plt.axis('off')
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_feature_importance(features, importances):
    """Create a horizontal bar plot of feature importances"""
    plt.figure(figsize=(8, 4))
    plt.barh(features, importances, color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_bias_heatmap(matrix, labels):
    """Create a heatmap for bias detection results"""
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap="coolwarm")
    plt.title("Bias Detection Heatmap")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def format_as_html_bullets(text):
    """Convert bullet point text to HTML format"""
    lines = text.split('\n')
    html = "<ul class='list-group list-group-flush mb-3'>"
    for line in lines:
        if line.strip().startswith('-') or line.strip().startswith('•'):
            html += f"<li class='list-group-item'>{line.strip()[1:].strip()}</li>"
        elif line.strip():
            html += f"<li class='list-group-item'>{line.strip()}</li>"
    html += "</ul>"
    return html

# ---------------- AGENT FUNCTIONS ----------------
def risk_scoring_agent(text):
    prompt = (
        "You are a formal loan risk analysis assistant. Carefully review the following loan document and assign a single numerical risk score from 1 (lowest risk) to 10 (highest risk), based on all available factors such as applicant's income stability, net worth, liabilities, collateral, guarantor, loan purpose, and any red flags. "
        "Base your scoring on current industry-standard credit risk assessment models. "
        "Respond ONLY with the final risk score as a single number out of 10\n\n"
        f"{text}"
    )
    score_text = query_granite(prompt)
    try:
        # Extract first integer found in response
        score = int(re.search(r'\d+', score_text).group())
        score = max(1, min(score, 10))  # Clamp between 1-10
        gauge_img = create_risk_gauge(score)
        html_output = f'''
        <div class="text-center mb-4">
            <img src="data:image/png;base64,{gauge_img}" class="img-fluid">
        </div>
        <p class="lead text-center">Risk Score: <span class="fw-bold">{score}/10</span></p>
        '''
    except:
        html_output = "<p class='text-danger'>Error: Could not parse risk score</p>"
    
    return {
        "summary": "Risk Score Analysis",
        "details": html_output
    }

def explainability_agent(text):
    prompt = (
        "You are an explainability agent for loan applications. Read the following loan document and provide a concise report of the key features influencing approval or rejection. "
        "List only the most important points as formal bullet points, covering:\n"
        "- Purpose of the Loan\n"
        "- Applicant Background\n"
        "- Net Worth\n"
        "- Guarantor or Collateral\n"
        "Do not include narrative or extra explanation—list only the essential findings as bullet points.\n\n"
        f"{text}"
    )
    response = query_granite(prompt)
    # Example features and importances (replace with dynamic data if available)
    features = ['Loan Purpose', 'Applicant Background', 'Net Worth', 'Guarantor']
    importances = [0.4, 0.3, 0.2, 0.1]  # Example values; adjust as needed
    feature_img = plot_feature_importance(features, importances)
    html_output = f'''
    <div class="mb-4">
        <img src="data:image/png;base64,{feature_img}" class="img-fluid">
    </div>
    {format_as_html_bullets(response)}
    '''
    return {
        "summary": "Explainability Report",
        "details": html_output
    }

def bias_detection_agent(text):
    prompt = (
        "You are a bias detection reviewer. Analyze the following loan application for potential bias and provide a concise summary. "
        "List only the key findings as formal bullet points, addressing:\n"
        "- Gender Bias\n"
        "- Location Bias\n"
        "- Income Bias\n"
        "- Overall Fairness\n"
        "Do not include narrative or extra explanation—list only the essential findings as bullet points.\n\n"
        f"{text}"
    )
    response = query_granite(prompt)
    # Example matrix (replace with dynamic data if available)
    labels = ['Gender', 'Location', 'Income', 'Fairness']
    matrix = np.array([
        [1, 0.2, 0.1, 0.3],
        [0.2, 1, 0.4, 0.5],
        [0.1, 0.4, 1, 0.6],
        [0.3, 0.5, 0.6, 1]
    ])
    heatmap_img = plot_bias_heatmap(matrix, labels)
    html_output = f'''
    <div class="mb-4">
        <img src="data:image/png;base64,{heatmap_img}" class="img-fluid">
    </div>
    {format_as_html_bullets(response)}
    '''
    return {
        "summary": "Bias Detection Report",
        "details": html_output
    }

# ---------------- MAIN ANALYSIS ----------------
def analyze_document(path):
    raw_text = extract_text_from_pdf(path)
    return {
        "Risk Scoring": risk_scoring_agent(raw_text),
        "Explainability": explainability_agent(raw_text),
        "Bias Detection": bias_detection_agent(raw_text)
    }
