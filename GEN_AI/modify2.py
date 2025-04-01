import google.generativeai as genai
import csv
import nltk
from nltk.corpus import stopwords
import re
import time

# Download stopwords if not already available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    # Reconstruct the cleaned text
    return ' '.join(filtered_words)

 
# Configure Gemini AI
genai.configure(api_key="AIzaSyAceKZACJb0Si9xZIVbwBL4dcS4veiAz34")
model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-1219")
 
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return ""
 
def ask_llm(content, questions):
    answers = {}
    cleaned_content = clean_text(content)
    for question in questions:
        time.sleep(3)
        prompt = f"""
        Given the following company information:
        {cleaned_content}
       
        Answer the question concisely:
        {question}
        """
    
        try:
            response = model.generate_content(prompt)
 
            # Check if the response is valid
            if response and response.candidates:
                answers[question] = response.text.strip() if response.text else "No valid response."
            else:
                answers[question] = "Response blocked or no valid output."
 
        except ValueError as e:
            if "finish_reason" in str(e):
                answers[question] = "Response blocked due to copyrighted content."
            else:
                answers[question] = f"Error: {e}"
 
        except Exception as e:
            answers[question] = f"Unexpected error: {e}"
 
    return answers
 
def main():
    questions = [
        "What is the company's mission statement or core values?",
        "What products or services does the company offer?",
        "When was the company founded, and who were the founders?",
        "Where is the company's headquarters located?",
        "Who are the key executives or leadership team members?",
        "Has the company received any notable awards or recognitions?"
    ]
 
    # Identify text files
    files = [
        'snap.txt','shopify.txt','squareup.txt',
        'stripe.txt','tesla.txt','dropbox.txt','hmgroup.txt','robinhood.txt','gsk.txt','spotify.txt']
 
    if not files:
        print("Error: No text files found.")
        return
 
    # CSV Output
    csv_file = "company_data1.csv"
   
    with open(csv_file, "a+", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
       
        # Write header row
        header = ["Website"] + questions
        writer.writerow(header)
 
        for file_name in files:
            content = read_text_file(file_name)
 
            if not content:
                continue
 
            answers = ask_llm(content, questions)
            row = [file_name.replace(".txt", "")] + [answers[q] for q in questions]
 
            writer.writerow(row)
 
    print(f"Data saved to {csv_file}")
 
if __name__ == "__main__":
    main()