import gradio as gr
import pandas as pd
import joblib
import pickle, gzip

from google.colab import drive
drive.mount('/content/drive')

# ===== LOAD TRAINED MODEL AND ENCODERS =====
rf_model = joblib.load("rf_model_small.pkl")  # Your trained Random Forest
label_encoders = joblib.load("label_encoders.pkl")  # Dict with encoders for 'last_view_cat', 'most_freq_cat', 'target_category'

# Features used during training (must match exactly)
feature_cols = [
    "last_view_cat",
    "most_freq_cat",
    "unique_cats_viewed",
    "total_views_before_cart"
]

# ===== PREDICTION FUNCTION =====
def recommend_category(last_view_cat, most_freq_cat, unique_cats_viewed, total_views_before_cart):
    """
    Predict the category of item the user is most likely to add to cart
    """
    # Encode categorical features
    last_view_encoded = label_encoders["last_view_cat"].transform([str(last_view_cat)])[0]
    most_freq_encoded = label_encoders["most_freq_cat"].transform([str(most_freq_cat)])[0]

    # Create DataFrame for prediction
    input_df = pd.DataFrame([[
        last_view_encoded,
        most_freq_encoded,
        unique_cats_viewed,
        total_views_before_cart
    ]], columns=feature_cols)

    # Predict top-1
    pred_encoded = rf_model.predict(input_df)[0]
    pred_category = label_encoders["target_category"].inverse_transform([pred_encoded])[0]

    # Predict top-3
    proba_df = pd.DataFrame(rf_model.predict_proba(input_df), columns=rf_model.classes_)
    top3 = proba_df.T.sort_values(by=0, ascending=False).head(3)
    top3_results = [
        (label_encoders["target_category"].inverse_transform([idx])[0], f"{prob*100:.2f}%")
        for idx, prob in zip(top3.index, top3[0])
    ]

    return pred_category, top3_results

# ===== GRADIO INTERFACE =====
with gr.Blocks() as demo:
    gr.Markdown("## 🛒 Product Category Recommendation System")
    gr.Markdown("Predict the most likely category a user will add to cart, based on their viewing behavior.")

    with gr.Row():
        last_view_cat = gr.Textbox(label="Last Viewed Category ID", placeholder="Enter category ID of last viewed item")
        most_freq_cat = gr.Textbox(label="Most Frequently Viewed Category ID", placeholder="Enter most frequent category ID viewed")

    with gr.Row():
        unique_cats_viewed = gr.Number(label="Number of Unique Categories Viewed", value=3)
        total_views_before_cart = gr.Number(label="Total Views Before Add-to-Cart", value=5)

    predict_btn = gr.Button("Predict Category")

    with gr.Row():
        output_category = gr.Textbox(label="Predicted Category", interactive=False)
        output_top3 = gr.JSON(label="Top 3 Predictions (Category, Probability)")

    predict_btn.click(
        fn=recommend_category,
        inputs=[last_view_cat, most_freq_cat, unique_cats_viewed, total_views_before_cart],
        outputs=[output_category, output_top3]
    )

if __name__ == "__main__":
    demo.launch()
