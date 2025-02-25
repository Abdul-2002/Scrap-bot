from flask import Flask, request, jsonify, send_from_directory
from bs4 import BeautifulSoup
import requests
from transformers import pipeline
from flask_cors import CORS
from urllib.parse import urlparse, urljoin
import time

app = Flask(__name__, static_folder='static')
CORS(app)

# Load a more accurate question-answering model
qa_pipeline = pipeline('question-answering', model="deepset/roberta-large-squad2")

# Load a summarization model for making answers more descriptive
summarizer = pipeline('summarization', model="facebook/bart-large-cnn")

# Define headers to mimic a real browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

session = requests.Session()
session.headers.update(HEADERS)

def scrape_website(url, depth=1):
    """Scrapes all relevant content from a website, handling pagination if necessary."""
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

    soup = BeautifulSoup(response.content, 'html.parser')
    elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div'])
    text = ' '.join([elem.get_text() for elem in elements]).strip()

    # Follow pagination if available
    if depth > 1:
        next_page = find_next_page_link(soup, url)
        if next_page:
            time.sleep(2)
            text += "\n\n" + scrape_website(next_page, depth - 1)

    return text if text else "No content found on the page."

def find_next_page_link(soup, base_url):
    """Finds the next page URL if pagination exists."""
    next_page = soup.find('a', text=lambda x: x and "next" in x.lower())
    if next_page and next_page.get('href'):
        return urljoin(base_url, next_page['href'])
    return None

@app.route('/scrape', methods=['POST'])
def scrape():
    data = request.get_json()
    urls = data.get('urls')
    depth = data.get('depth', 1)

    if not urls or not isinstance(urls, list):
        return jsonify({'error': 'A list of URLs is required'}), 400

    combined_text = ""

    for url in urls:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return jsonify({'error': f'Invalid URL: {url}'}), 400

        scraped_text = scrape_website(url, depth=depth)
        combined_text += scraped_text + "\n\n"

    return jsonify({'text': combined_text[:50000]})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    context = data.get('context')

    if not question or not context:
        return jsonify({'error': 'Both question and context are required'}), 400

    try:
        response = qa_pipeline({'question': question, 'context': context})
        raw_answer = response.get('answer', 'No answer found.')
        confidence = response.get('score', 0.0)

        # Generate a more descriptive answer using summarization
        expanded_answer = summarize_answer(raw_answer, context)

        detailed_response = {
            'question': question,
            'answer': expanded_answer,
            'confidence': f"{confidence * 100:.2f}%",
            'source': 'Extracted from the provided context',
            'context_used': context[:1000] + "..." if len(context) > 1000 else context
        }

        return jsonify(detailed_response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def summarize_answer(answer, context):
    """
    Enhances the original answer by summarizing relevant context around it.
    """
    full_text = f"Answer: {answer}. Additional Information: {context[:3000]}"  # Limit context size
    summary = summarizer(full_text, max_length=100, min_length=50, do_sample=False)
    
    return summary[0]['summary_text'] if summary else answer

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
