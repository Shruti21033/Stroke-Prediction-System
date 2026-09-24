import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================
# 1. LOAD DATASET
# ============================================

data = pd.read_csv("stroke_dataset.csv")

print("Dataset loaded successfully")
print("Dataset shape:", data.shape)


# ============================================
# 2. REMOVE ID COLUMN
# ============================================

if "id" in data.columns:
    data = data.drop("id", axis=1)


# ============================================
# 3. DEFINE FEATURES AND TARGET
# ============================================

X = data.drop("stroke", axis=1)
y = data["stroke"]


# ============================================
# 4. IDENTIFY COLUMNS
# ============================================

categorical_columns = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

numeric_columns = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi"
]


# ============================================
# 5. NUMERIC PIPELINE
# ============================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ============================================
# 6. CATEGORICAL PIPELINE
# ============================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ============================================
# 7. PREPROCESSING
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ]
)


# ============================================
# 8. MACHINE LEARNING MODEL
# ============================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ============================================
# 9. COMPLETE PIPELINE
# ============================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================
# 10. TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================
# 11. TRAIN MODEL
# ============================================

print("Training model...")

pipeline.fit(X_train, y_train)


# ============================================
# 12. TEST MODEL
# ============================================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================
# 13. SAVE MODEL
# ============================================

joblib.dump(
    pipeline,
    "stroke_pipeline.pkl"
)

print("\nModel saved successfully!")
print("File created: stroke_pipeline.pkl")