import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

# -----------------------
# 1. Load Data
# -----------------------
df = pd.read_csv("feature_full.csv") 
X = df.drop(columns=['label','update_name'])
print(X.columns)
y = df['label']

# -----------------------
# 2. Variance Threshold
# -----------------------
var_thresh = 0.01
vt = VarianceThreshold(threshold=var_thresh)
X_vt = vt.fit_transform(X)
retained_features = X.columns[vt.get_support()]
X_vt_df = pd.DataFrame(X_vt, columns=retained_features)

print(f"[INFO] Features after Variance Threshold: {list(retained_features)}, number of features: {len(retained_features)}")

# -----------------------
# 3. Mutual Information
# -----------------------
mi = mutual_info_classif(X_vt_df, y, discrete_features=False)
mi_series = pd.Series(mi, index=X_vt_df.columns).sort_values(ascending=False)

# Plot MI scores
plt.figure(figsize=(10, 5))
sns.barplot(x=mi_series.values, y=mi_series.index, hue=mi_series.index, palette="viridis", legend=False)
plt.title("Mutual Information Scores")
plt.xlabel("Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("figs/mi_scores.png")
plt.show()


# Select top-k features based on MI
mi_threshold = 0.1
selected_features = mi_series[mi_series > mi_threshold].index.tolist()
X_mi = X_vt_df[selected_features]
X_mi_with_label = X_mi.copy()
X_mi_with_label['label'] = y
X_mi_with_label.to_csv("filtered_features.csv", index=False)


print(f"[INFO] Features with MI > {mi_threshold}: {selected_features}, number of features: {len(selected_features)}")

plt.figure(figsize=(10, 5))
sns.barplot(x=mi_series[selected_features].values, y=selected_features, hue=selected_features, palette="viridis", legend=False)
plt.title("Mutual Information Scores for Selected Features")
plt.xlabel("Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("figs/mi_scores_selected_features.png")
plt.show()
