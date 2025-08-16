# 🛒 Product Category Recommendation System

This project builds and deploys a **machine learning–based recommendation system** that predicts the most likely product category a user will add to their cart, based on their **viewing behavior**. The system uses a **Random Forest Classifier**, optimized for both **Top-1 and Top-3 accuracy**, and is deployed with **Gradio** for interactive testing.

---

## 🚀 Features

* Predicts **Top-1** and **Top-3 most likely categories** for user add-to-cart actions.
* Uses **user interaction features** (last viewed category, most frequent category, unique categories viewed, total views before cart).
* Deployed via **Gradio Interface** for easy testing and demonstration.
* Compressed model (`rf_model_small.joblib`) for lightweight deployment (<100MB).
* Includes saved **Label Encoders** for consistent category mapping.

---

## 📂 Project Structure

```
├── app.py                  # Gradio app for deployment  
├── rf_model_small.joblib   # Trained Random Forest model (compressed)  
├── label_encoders.pkl      # Encoders for categorical features  
├── requirements.txt        # Dependencies  
├── README.md               # Project documentation  
└── data/                   # (Optional) Dataset or sample inputs  
```

---

## ⚙️ Setup & Installation (Running Locally)

1. Clone this repository:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
   cd <your-space-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run locally:
   ```bash
   python app.py
   ```
---

You can also visit this link to test the system:
https://huggingface.co/spaces/selorm-etse-forfoe/ecommerce-item-category-recommendation-system

## 🎯 Usage

When launched, the app will display an interface with the following inputs:

1. **Last Viewed Category ID** → Category ID of the last item viewed by the user.
2. **Most Frequently Viewed Category ID** → Category ID that the user viewed most often.
3. **Number of Unique Categories Viewed** → Count of distinct categories viewed before adding to cart.
4. **Total Views Before Add-to-Cart** → Total number of items viewed before cart action.

**Outputs:**

1. **Predicted Category** → The most likely category the user will add to cart.
2. **Top 3 Predictions** → Ranked list of top-3 predicted categories with probabilities.

---

## 📊 Model Performance

1. **Top-1 Accuracy:** \~85%
2. **Top-3 Accuracy:** \~92%
3. **Macro Precision:** \~0.51
4. **Macro Recall:** \~0.39
5. **Macro F1-score:** \~0.42

---

## 🔑 Key Insights

1. Viewing patterns strongly influence add-to-cart predictions.
2. Engagement peaks midweek evenings (Wed–Thu, 7–11 PM).
3. Some categories dominate conversions, while others rarely convert.
4. Abnormal users (e.g., bots) can skew results → anomaly detection is needed.

---

## 📝 Recommendations

1. Deploy the **recommendation engine** into production.
2. Improve **funnel efficiency** (reduce checkout friction, address pricing barriers).
3. Leverage **midweek campaigns** to maximize engagement.
4. Monitor and mitigate **abnormal user behavior**.
5. Enhance the dataset with **category names and metadata** for richer insights.

---

## 👨‍💻 Author

**Selorm Etse-Forfoe**
[selorm.etse5@gmail.com](mailto:selorm.etse5@gmail.com)
Ghana

---
