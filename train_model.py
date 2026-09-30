"""
train_model.py
---------------
Devoir - Deploiement de Projet IA (Pr. Kalloubi)

Etapes :
1. Choix d'un dataset tabulaire (Diabetes dataset - scikit-learn / UCI)
2. Preparation des donnees (valeurs manquantes)
3. Analyse des caracteristiques les plus pertinentes
4. Comparaison de plusieurs modeles ML
5. Sauvegarde du meilleur modele en .pkl pour l'application Flask

Dataset : "Diabetes" (Efron et al., 2004 - reference standard scikit-learn / UCI)
442 patients, 10 variables physiologiques, cible = mesure quantitative de la
progression de la maladie un an apres la mesure de reference.
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Chargement du dataset (valeurs reelles, non standardisees)
# ---------------------------------------------------------------------------
print("Chargement du dataset...")
data = load_diabetes(as_frame=True, scaled=False)
df = data.frame

print(f"Dimensions : {df.shape}")
print(df.head())


# ---------------------------------------------------------------------------
# 2. Preparation des donnees - traitement des valeurs manquantes
# ---------------------------------------------------------------------------
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

# Le dataset ne contient aucune valeur manquante, mais on garde ce traitement
# generique pour respecter la consigne du TP et rendre le script reutilisable
# avec un autre dataset qui en contiendrait.
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

assert df.isnull().sum().sum() == 0, "Il reste des valeurs manquantes !"
print("Aucune valeur manquante restante.")


# ---------------------------------------------------------------------------
# 3. Caracteristiques les plus pertinentes
# ---------------------------------------------------------------------------
print("\nCorrelation de chaque variable avec la cible (target) :")
correlations = df.corr(numeric_only=True)["target"].drop("target").sort_values(
    key=abs, ascending=False
)
print(correlations)

# On ne garde que les variables les plus correlees a la cible (|r| > 0.39) :
# bmi, s5 (log triglycerides), bp, s4 et s3 ressortent nettement devant les
# autres (age, sexe, s1, s2, s6 ont une correlation trop faible pour etre
# vraiment utiles). Cela simplifie aussi le formulaire de prediction.
FEATURES = ["bmi", "s5", "bp", "s4", "s3"]
TARGET = "target"

print(f"\nCaracteristiques retenues (|correlation| > 0.39) : {FEATURES}")

X = df[FEATURES]
y = df[TARGET]


# ---------------------------------------------------------------------------
# 4. Comparaison de plusieurs modeles ML
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

results = []
best_model = None
best_model_name = None
best_r2 = -np.inf

print("\n--- Comparaison des modeles ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    train_r2 = model.score(X_train, y_train)

    results.append(
        {"model": name, "train_r2": train_r2, "test_r2": r2, "rmse": rmse}
    )
    print(f"{name:20s} | Train R2: {train_r2:.3f} | Test R2: {r2:.3f} | RMSE: {rmse:.2f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

results_df = pd.DataFrame(results).sort_values("test_r2", ascending=False)
print("\nClassement final :")
print(results_df.to_string(index=False))

print(f"\nMeilleur modele retenu : {best_model_name} (Test R2 = {best_r2:.3f})")


# ---------------------------------------------------------------------------
# 5. Sauvegarde du meilleur modele
# ---------------------------------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model_info.txt", "w") as f:
    f.write(f"Modele retenu : {best_model_name}\n")
    f.write(f"Test R2 : {best_r2:.3f}\n")
    f.write(f"Features (dans l'ordre attendu par le modele) : {FEATURES}\n")

print("\nmodel.pkl genere avec succes dans le dossier courant.")
