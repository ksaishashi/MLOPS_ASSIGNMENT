import pickle
from sklearn.metrics import roc_auc_score

def test_model_loads(trained_model_path):
    with open(trained_model_path, "rb") as f:
        model = pickle.load(f)

    assert model is not None

def test_model_predict(processed_df, trained_model_path):
    with open(trained_model_path, "rb") as f:
        model = pickle.load(f)

    X = processed_df.drop(columns=["target"])
    preds = model.predict(X)

    assert len(preds) == len(X)

def test_model_predict_proba(processed_df, trained_model_path):
    with open(trained_model_path, "rb") as f:
        model = pickle.load(f)

    X = processed_df.drop(columns=["target"])
    probs = model.predict_proba(X)[:, 1]

    assert ((probs >= 0) & (probs <= 1)).all()

def test_model_roc_auc(processed_df, trained_model_path):
    with open(trained_model_path, "rb") as f:
        model = pickle.load(f)

    X = processed_df.drop(columns=["target"])
    y = processed_df["target"]

    probs = model.predict_proba(X)[:, 1]
    score = roc_auc_score(y, probs)

    assert score >= 0.5
