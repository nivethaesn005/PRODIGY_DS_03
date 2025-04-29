import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

file_path = r"D:\prodigy_Task-03\bank-additional\bank-additional\bank-additional-full.csv"
data = pd.read_csv(file_path, sep=';')

X = data.drop("y", axis=1)
y = data["y"].map({"no": 0, "yes": 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer([("cat", OneHotEncoder(drop='first', sparse_output=False), categorical_cols),("num", StandardScaler(), numeric_cols)])

model_pipeline = Pipeline([("preprocess", preprocessor),("clf", DecisionTreeClassifier(max_depth=6, random_state=42))])

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("\n🎓 MODEL EVALUATION\n" + "-"*30)
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy Score: {accuracy_score(y_test, y_pred):.4f}\n")

# 🔥 Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

final_model = model_pipeline.named_steps["clf"]
onehot_features = model_pipeline.named_steps["preprocess"].transformers_[0][1].get_feature_names_out(categorical_cols)
all_features = np.concatenate([onehot_features, numeric_cols])
importances = final_model.feature_importances_

feat_imp_df = pd.DataFrame({"Feature": all_features,"Importance": importances}).sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_imp_df, color="skyblue")
plt.title("Top 10 Important Features in Decision Tree")
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.show()

job_target = pd.crosstab(data['job'], data['y'], normalize='index') * 100
job_target.sort_values(by='yes', ascending=False, inplace=True)

job_target.plot(kind='barh', stacked=True, color=['salmon', 'skyblue'], figsize=(10, 7))
plt.title("Purchase Distribution (%) by Job Title")  # Removed emoji to avoid font error
plt.xlabel("Percentage")
plt.legend(title='Subscribed')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\n📌 TOP 10 IMPORTANT FEATURES")
print(feat_imp_df.to_string(index=False))
