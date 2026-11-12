from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


APP_DIR = Path(__file__).resolve().parent
SENTIMENT_TRAINING_EXAMPLES = (
    ("The report is bad.", "negative"),
    ("This is a bad report.", "negative"),
    ("The care I received was bad and disappointing.", "negative"),
    ("The service was bad and I felt ignored.", "negative"),
    ("The report is good.", "positive"),
    ("This is a good report.", "positive"),
    ("The care I received was good and reassuring.", "positive"),
    ("The service was good and I felt supported.", "positive"),
    ("The report was added to my patient file.", "neutral"),
    ("I received a copy of the report.", "neutral"),
    ("The report lists my appointment details.", "neutral"),
    ("The clinician reviewed the report during my visit.", "neutral"),
)
DATASETS = {
    "Patient sentiment": {
        "data": "synthetic_patient_sentiment.csv",
        "label": "sentiment",
        "description": "Estimate the tone expressed in a synthetic patient comment.",
    },
    "Medical document": {
        "data": "synthetic_medical_document_classification.csv",
        "label": "category",
        "description": "Classify a synthetic medical abstract by its topic.",
    },
}


st.set_page_config(
    page_title="Healthcare NLP Studio",
    page_icon="🩺",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Training and evaluating the text classifier...")
def load_classifier(
    path: str, label_column: str
) -> tuple[Pipeline, tuple[str, ...], float]:
    data = pd.read_csv(path)
    if "text" not in data.columns or label_column not in data.columns:
        raise ValueError(
            f"{Path(path).name} must contain 'text' and '{label_column}' columns."
        )

    data = data[["text", label_column]].dropna().copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    if data.empty:
        raise ValueError(f"{Path(path).name} contains no usable text rows.")

    train_text, test_text, train_labels, test_labels = train_test_split(
        data["text"],
        data[label_column].astype(str),
        test_size=0.20,
        random_state=42,
        stratify=data[label_column].astype(str),
    )

    def build_pipeline() -> Pipeline:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 3),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, C=4, random_state=42),
                ),
            ]
        )

    evaluation_model = build_pipeline()
    evaluation_model.fit(train_text, train_labels)
    holdout_accuracy = float(evaluation_model.score(test_text, test_labels))

    final_text = data["text"]
    final_labels = data[label_column].astype(str)
    if label_column == "sentiment":
        examples = pd.DataFrame(
            SENTIMENT_TRAINING_EXAMPLES,
            columns=["text", label_column],
        )
        final_text = pd.concat([final_text, examples["text"]], ignore_index=True)
        final_labels = pd.concat(
            [final_labels, examples[label_column]],
            ignore_index=True,
        )

    classifier = build_pipeline()
    classifier.fit(final_text, final_labels)
    classes = tuple(str(label) for label in classifier.classes_)
    return classifier, classes, holdout_accuracy


def show_dataset_summary(dataset: pd.DataFrame, label_column: str) -> None:
    missing = [column for column in ("text", label_column) if column not in dataset]
    if missing:
        st.error(f"Dataset is missing required columns: {', '.join(missing)}.")
        st.stop()

    usable = dataset[["text", label_column]].dropna()
    class_count = usable[label_column].nunique()
    first, second, third = st.columns(3)
    first.metric("Synthetic examples", f"{len(usable):,}")
    second.metric("Available classes", class_count)
    third.metric("Classifier", "TF-IDF + logistic regression")


st.title("Healthcare NLP Studio")
st.caption("Explore two text classifiers trained on synthetic examples.")
st.warning(
    "Educational demonstration only. The datasets are synthetic; predictions are "
    "not medical advice and must not be used for diagnosis or care decisions."
)

with st.sidebar:
    st.header("Choose a task")
    task = st.radio("Classifier", tuple(DATASETS))
    st.divider()
    st.caption(
        "Do not enter real patient information. Text is processed locally by this app."
    )

config = DATASETS[task]
dataset_path = APP_DIR / config["data"]

if not dataset_path.is_file():
    st.error(f"Dataset not found: {dataset_path.name}.")
    st.stop()

dataset = load_dataset(str(dataset_path))
show_dataset_summary(dataset, config["label"])

st.subheader(task)
st.write(config["description"])
default_text = (
    "The doctor was kind and explained everything clearly."
    if task == "Patient sentiment"
    else "This study examines asthma symptoms, lung function, and respiratory infections."
)
text = st.text_area(
    "Enter a synthetic example or other non-sensitive text",
    value=default_text,
    height=150,
)

if st.button("Analyze text", type="primary"):
    text = text.strip()
    if not text:
        st.warning("Enter some text before running the classifier.")
    else:
        try:
            classifier, classes, holdout_accuracy = load_classifier(
                str(dataset_path), config["label"]
            )
        except ValueError as error:
            st.error(str(error))
            st.stop()

        vectorizer = classifier.named_steps["tfidf"]
        if vectorizer.transform([text]).nnz == 0:
            st.warning(
                "This text contains no words the classifier learned from its "
                "training data, so its prediction is not meaningful."
            )
        probabilities = np.asarray(classifier.predict_proba([text])[0])
        if probabilities.size != len(classes):
            st.error(
                "The model output does not match the dataset's class labels. "
                "Check that the classifier and CSV use the same labels."
            )
            st.stop()

        predicted_index = int(np.argmax(probabilities))
        st.success(f"Predicted {config['label']}: **{classes[predicted_index]}**")
        st.metric(
            "Model confidence",
            f"{probabilities[predicted_index]:.1%}",
            help="Estimated probability from the classifier; it is not a guarantee.",
        )
        st.caption(
            f"Holdout accuracy on synthetic examples: {holdout_accuracy:.1%}. "
            "Confidence is the model's estimated probability, not a guarantee."
        )
        chart = pd.DataFrame(
            {"Probability": probabilities},
            index=pd.Index(classes, name="Class"),
        )
        st.bar_chart(chart, y="Probability", height=280)

with st.expander("About the data and model"):
    st.write(
        "The saved LSTM models in this project remained at chance-level accuracy "
        "and returned the same probabilities for every input. This app instead "
        "trains a TF-IDF and logistic-regression classifier from the included "
        "synthetic CSV. For sentiment, it also adds a small balanced set of "
        "labeled synthetic examples for common positive, negative, and neutral "
        "phrases. The reported accuracy is measured on a separate holdout split "
        "from the CSV before those extra examples are added."
    )
