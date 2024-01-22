import tkinter as tk
from tkinter import filedialog, scrolledtext
import nltk
import spacy
import json
import os
import datetime
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Setup and downloads
def setup():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    return spacy.load('en_core_web_sm')


def parse_line(line):
    try:
        timestamp, text = line.split("|", 1)
        timestamp = datetime.datetime.strptime(timestamp.strip(), "%Y-%m-%d %H:%M:%S")
        return timestamp, text.strip()
    except ValueError:
        return None, line.strip()


def tokenize_and_clean(text, stop_words):
    words = word_tokenize(text.lower())
    return [word for word in words if word not in stop_words and word.isalpha()]

# Categorize issues based on keywords
def categorize_issues(text, issue_keywords):
    categories = Counter()
    for category, keywords in issue_keywords.items():
        if any(keyword in text for keyword in keywords):
            categories.update([category])
    return categories


def generate_ngrams(words, n):
    return zip(*[words[i:] for i in range(n)])

# Main analysis function
def analyze_text(file_path, nlp, ngram_size=2):
    stop_words = set(stopwords.words('english'))
    word_freq, ngram_freq, entity_freq = Counter(), Counter(), Counter()
    question_freq, resolutions, issue_categories = Counter(), Counter(), Counter()
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_time_series = []

    resolution_phrases = ["resolved", "will get back to you", "solution", "fixed"]
    issue_keywords = {
        'billing': ['invoice', 'billing', 'charge', 'payment'],
        'technical': ['error', 'problem', 'bug', 'issue'],
        'support': ['help', 'support', 'service', 'assistance']
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                timestamp, text = parse_line(line)
                words = tokenize_and_clean(text, stop_words)
                word_freq.update(words)
                ngram_freq.update(generate_ngrams(words, ngram_size))
                entity_freq.update([ent.text for ent in nlp(text).ents])
                question_freq.update([text] if "?" in text else [])
                sentiment = sentiment_analyzer.polarity_scores(text)['compound']
                if timestamp:
                    sentiment_time_series.append((timestamp.isoformat(), sentiment))
                if any(phrase in text for phrase in resolution_phrases):
                    resolutions.update([text])
                issue_categories.update(categorize_issues(text, issue_keywords))

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None

    return word_freq, ngram_freq, entity_freq, question_freq, sentiment_time_series, resolutions, issue_categories

# Save results to JSON
def save_to_json(data, output_file, save_n):
    # Convert tuple keys to strings
    def ngram_to_string(ngram):
        return '_'.join(ngram)

    limited_data = {
        'word_freq': dict(data[0].most_common(save_n)),
        'ngram_freq': {ngram_to_string(k): v for k, v in data[1].most_common(save_n)},
        'entity_freq': dict(data[2].most_common(save_n)),
        'question_freq': dict(data[3].most_common(save_n)),
        'sentiment_time_series': data[4][:save_n],
        'resolutions': dict(data[5].most_common(save_n)),
        'issue_categories': dict(data[6])
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(limited_data, f, ensure_ascii=False, indent=4)
    print(f"JSON file saved successfully at {output_file}")

# GUI
def setup_ui():
    window = tk.Tk()
    window.title("Text Analysis Tool")

    # File selection
    def select_file():
        file_path.set(filedialog.askopenfilename())

    tk.Label(window, text="Select File:").grid(row=0, column=0)
    file_path = tk.StringVar()
    tk.Entry(window, textvariable=file_path).grid(row=0, column=1)
    tk.Button(window, text="Browse", command=select_file).grid(row=0, column=2)

    tk.Label(window, text="Output JSON File Path:").grid(row=1, column=0)
    output_file = tk.StringVar()
    tk.Entry(window, textvariable=output_file).grid(row=1, column=1)

    tk.Label(window, text="Number of Top Results for Analysis:").grid(row=2, column=0)
    top_n = tk.IntVar(value=10)  
    tk.Entry(window, textvariable=top_n).grid(row=2, column=1)

    tk.Label(window, text="Number of Results to Save:").grid(row=3, column=0)
    save_n = tk.IntVar(value=10) 
    tk.Entry(window, textvariable=save_n).grid(row=3, column=1)

    results_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=160, height=50)
    results_area.grid(row=5, column=0, columnspan=3)

    def analyze():
        nlp = setup()
        results = analyze_text(file_path.get(), nlp, top_n.get())
        if results:
            word_freq, ngram_freq, entity_freq, question_freq, sentiment_time_series, resolutions, issue_categories = results
            results_text = f"Word Frequencies: {dict(word_freq.most_common(top_n.get()))}\n"
            results_text += f"Ngram Frequencies: {[' '.join(k) for k, v in ngram_freq.most_common(top_n.get())]}\n"
            results_text += f"Named Entities: {dict(entity_freq.most_common(top_n.get()))}\n"
            results_text += f"Questions: {dict(question_freq.most_common(top_n.get()))}\n"
            results_text += f"Issue Categories: {dict(issue_categories)}\n"
            results_area.delete('1.0', tk.END)
            results_area.insert(tk.INSERT, results_text)
            save_to_json(results, output_file.get(), save_n.get())
        else:
            results_area.delete('1.0', tk.END)
            results_area.insert(tk.INSERT, "Analysis was not completed due to an error or file not found.")

    tk.Button(window, text="Analyze", command=analyze).grid(row=4, column=0)
    tk.Button(window, text="Exit", command=window.quit).grid(row=4, column=2)

    window.mainloop()

if __name__ == "__main__":
    setup_ui()
