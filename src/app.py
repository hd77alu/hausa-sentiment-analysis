import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from src.prediction import predict_hausa_sentiment

def process_input(user_text):
    confidences, main_label = predict_hausa_sentiment(user_text)
    return confidences, main_label

# Build Gradio UI
demo = gr.Interface(
    fn=process_input,
    inputs=gr.Textbox(
        lines=3, 
        placeholder="Shigar da rubutun Hausa a nan... (Enter Hausa text here)",
        label="Hausa Text Input"
    ),
    outputs=[
        gr.Label(num_top_classes=3, label="Class Probabilities"),
        gr.Textbox(label="Prediction")
    ],
    title="Hausa Sentiment Analysis Classifier",
    description="This is a simple web application that allows you to input Hausa text and receive sentiment predictions (negative, neutral, positive).",
    examples=[
        ["Muna murna sosai da wannan sakamako mai kyau!"],
        ["Wannan tsari bai yi kyau ko kadan ba."],
        ["An gudanar da taron manema labarai a garin Kano a yau."]
    ],
)

if __name__ == "__main__":
    # Exposes built-in API endpoints at /api/predict
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
        theme="soft",
    )