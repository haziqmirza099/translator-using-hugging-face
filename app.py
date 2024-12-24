from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("C:/Users/haziq.mirza/Desktop/translator/token")
model = AutoModelForSeq2SeqLM.from_pretrained("C:/Users/haziq.mirza/Desktop/translator/model")

# Route for the homepage (UI with the form)
@app.route('/', methods=['GET', 'POST'])
def index():
    translated_text = ""
    if request.method == 'POST':
        text = request.form['input_text']
        src_lang = request.form['src_lang']
        tgt_lang = request.form['tgt_lang']

        # Translate the text
        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors="pt")
        inputs["forced_bos_token_id"] = tokenizer.lang_code_to_id[tgt_lang]
        
        outputs = model.generate(**inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Render the HTML template and pass the translated text
    return render_template('index.html', translated_text=translated_text)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

