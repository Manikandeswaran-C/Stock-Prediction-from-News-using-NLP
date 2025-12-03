
import gradio as gr
import joblib
import re
import os


os.environ["TRANSFORMERS_CACHE"] = "D:/event-stock-prediction/hf_cache"


model = joblib.load("models/stock_movement_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


summarizer = None
def get_summarizer():
    global summarizer
    if summarizer is None:
        from transformers import pipeline
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-6-6",  # smaller + faster
            cache_dir="D:/event-stock-prediction/hf_cache",
            device=-1  # CPU only
        )
    return summarizer


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentiment_analyzer = SentimentIntensityAnalyzer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def summarise_with_fallback(full_text: str):
    if not full_text.strip():
        return ""
    try:
        s = get_summarizer()(
            full_text,
            max_length=40,
            min_length=15,
            do_sample=False,
            truncation=True
        )[0]["summary_text"]

      
        words = s.split()
        if len(words) > 40:
            s = " ".join(words[:40]) + "..."
    except Exception:
        s = " ".join(full_text.split()[:40]) + "..."
    return s


label_map = {
    -1: "📉 Stock will go DOWN",
     0: "😐 Stock NEUTRAL",
     1: "📈 Stock will go UP"
}


def predict(headline, description, use_sentiment):
    if not headline.strip() and not description.strip():
        return "", "", "⚠️ Please enter at least a headline or description.", ""

    combined_text = (headline + " " + description).strip()


    summary = summarise_with_fallback(combined_text)


    clean = clean_text(summary)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]

  
    try:
        proba = model.predict_proba(vec)[0]
        confidence = f"Confidence → DOWN: {proba[0]:.2f}, NEUTRAL: {proba[1]:.2f}, UP: {proba[2]:.2f}"
    except Exception:
        confidence = "⚠️ Confidence not available (model has no predict_proba)."

    if use_sentiment:
        sentiment_score = sentiment_analyzer.polarity_scores(summary)["compound"]
        if sentiment_score > 0.2:
            pred = 1
        elif sentiment_score < -0.2:
            pred = -1
        else:
            pred = 0
        confidence += f"\n(Sentiment override applied → score={sentiment_score:.2f})"

    return headline, summary, label_map.get(pred, "Unknown"), confidence


with gr.Blocks(theme="gradio/soft") as demo:
    gr.Markdown("## 📊 Stock Movement Predictor with Summarization")
    gr.Markdown("Enter a news headline and optional description. The system will summarize, predict stock movement, and show confidence.")

    with gr.Row():
        with gr.Column():
            headline = gr.Textbox(label="📰 Enter Headline", placeholder="Enter headline here")
            description = gr.Textbox(label="📄 Enter Description", placeholder="Enter description (optional)", lines=6)
            use_sentiment = gr.Checkbox(label="⚡ Use Sentiment Override (experimental)", value=False)
            btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear")

        with gr.Column():
            orig_out = gr.Textbox(label="📜 Original Headline", interactive=False)
            summary_out = gr.Textbox(label="📝 Summary", interactive=False, lines=6)
            prediction_out = gr.Textbox(label="📊 Prediction", interactive=False)
            confidence_out = gr.Textbox(label="📈 Confidence Details", interactive=False, lines=3)

    btn.click(predict, inputs=[headline, description, use_sentiment],
              outputs=[orig_out, summary_out, prediction_out, confidence_out])
    clear_btn.click(lambda: ("", "", "", ""), outputs=[headline, description, orig_out, summary_out])


if __name__ == "__main__":
    demo.launch()
