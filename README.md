# Healthcare NLP Studio

A Streamlit interface for two text classification tasks:

- Patient comment sentiment (`positive`, `neutral`, or `negative`)
- Medical document topic (`cardiology`, `endocrinology`, `neurology`, `oncology`, or `respiratory`)

## Run locally

From this folder, install dependencies and start Streamlit:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app expects both synthetic CSV datasets to remain alongside `app.py`. It
trains a TF-IDF and logistic-regression classifier from the selected dataset,
shows holdout accuracy, and uses the full dataset for its final predictions.
The sentiment classifier adds a small balanced set of synthetic examples for
common positive, negative, and neutral phrasing; these examples are not included
in the reported CSV holdout score.
The `.keras` files are not used by the app: the saved LSTM models did not learn
from their training data and returned the same probabilities for every input.

## Intended use

The bundled data is synthetic and for educational demonstration only. Do not
enter real patient information. Its holdout score does not indicate real-world
medical performance. Predictions are not medical advice and must not be used
for diagnosis or care decisions.

The original notebook requires TensorFlow to run its LSTM training cells;
TensorFlow is not needed to run the Streamlit app.
