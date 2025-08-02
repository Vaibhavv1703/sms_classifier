import gradio as gr
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def classify_sms(message):
    vect_msg = vectorizer.transform([message])
    prediction = model.predict(vect_msg)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Gradio UI
interface = gr.Interface(
    fn=classify_sms,
    inputs=gr.Textbox(label="Enter SMS Message"),
    outputs=gr.Label(label="Prediction"),
    title="SMS Spam Classifier",
    description="Enter a message to check if it's spam or not."
)

interface.launch()
