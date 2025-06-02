import pandas as pd
from transformers import pipeline
import gradio as gr

# Load Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline(model='distilbert-base-uncased-finetuned-sst-2-english')

# Core logic for batch prediction
def analyze_csv(file, text_column):
    # Read uploaded CSV file
    df = pd.read_csv(file)

    if text_column not in df.columns:
        return "Error: Column not found in file."

    texts = df[text_column].astype(str).tolist()
    results = sentiment_pipeline(texts)

    # Add results to the DataFrame
    df["Sentiment"] = [res["label"] for res in results]
    df["Confidence"] = [round(res["score"], 3) for res in results]

    # Save to a new CSV for download
    output_path = "sentiment_output.csv"
    df.to_csv(output_path, index=False)

    return output_path

# Gradio UI
def get_column_names(file):
    try:
        df = pd.read_csv(file.name, nrows=1)
        return list(df.columns)
    except:
        return []

with gr.Blocks() as app:
    gr.Markdown("## 📊 Batch Sentiment Analyzer (Hugging Face Transformers)")
    gr.Markdown("Upload a CSV file and select the column with the text (e.g., reviews, feedback).")

    with gr.Row():
        file_input = gr.File(label="Upload CSV", type="filepath")
        column_selector = gr.Dropdown(label="Select Text Column", choices=[], interactive=True)

    load_columns_btn = gr.Button("🔄 Load Columns")
    load_columns_btn.click(fn=get_column_names, inputs=file_input, outputs=column_selector)

    submit_btn = gr.Button("🚀 Run Sentiment Analysis")
    output_file = gr.File(label="📥 Download Output")

    submit_btn.click(fn=analyze_csv, inputs=[file_input, column_selector], outputs=output_file)

# Launch app
app.launch()
