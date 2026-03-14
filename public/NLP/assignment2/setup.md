# Setup Instructions

1. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate        # Mac/Linux
   .venv\Scripts\activate           # Windows

2. Install dependencies:
   pip install -r requirements.txt

3. Download the spaCy English model:
   python -m spacy download en_core_web_sm

4. Run the parser:
   python parser.py

5. To change the sentence, edit the bottom of parser.py:
   sentence = "your sentence here"
