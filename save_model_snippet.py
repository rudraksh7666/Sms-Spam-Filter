# Add this cell to the END of your Colab notebook, after training MultinomialNB
import joblib

# Replace these with your actual variable names from the notebook
# e.g. your CountVectorizer/TF-IDF object and your trained MultinomialNB model
joblib.dump(vectorizer, "spam_vectorizer.pkl")   # <- your vectorizer variable
joblib.dump(nb_model, "spam_model.pkl")          # <- your MultinomialNB variable

# Download both files to your machine (Colab: files sidebar -> right-click -> Download,
# or use the code below)
from google.colab import files
files.download("spam_vectorizer.pkl")
files.download("spam_model.pkl")
