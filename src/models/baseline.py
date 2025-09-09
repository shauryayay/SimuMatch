from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib


NUMERIC_FEATURES = [
"age", "avg_run_pace", "vdot_est", "num_events",
"distance_km", "target_pace", "diff_pace", "elevation_gain_m"
]
CATEGORICAL_FEATURES = ["gender"]




def create_pipeline():
preprocessor = ColumnTransformer(
transformers=[
("num", StandardScaler(), NUMERIC_FEATURES),
("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
]
)
model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
return pipe




def train_and_save(X, y, out_path="models/baseline_model.joblib"):
pipe = create_pipeline()
pipe.fit(X, y)
joblib.dump(pipe, out_path)
print(f"Saved model pipeline to {out_path}")
return pipe




def load_model(path="models/baseline_model.joblib"):
return joblib.load(path)
