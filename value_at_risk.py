"""
VALUE AT RISK (VaR) - GESTION DU RISQUE DE MARCHE
Auteur : Aymane El Adlouni

Ce programme calcule la Value at Risk d'un portefeuille d'actions, la mesure
de reference du risque de marche dans les salles de marches et les banques.

La VaR repond a une question simple et cruciale : dans le pire des cas, combien
puis-je perdre sur mon portefeuille sur un horizon donne ? Une VaR a 95% de
100 000 dirhams signifie qu'il y a 95% de chances de ne pas perdre plus de
100 000 dirhams sur la periode. C'est un calcul impose quotidiennement aux
banques par la reglementation de Bale.

Le programme calcule la VaR par trois methodes complementaires, puis la CVaR
qui mesure la perte moyenne au-dela de la VaR, et visualise la distribution
des pertes possibles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


# Portefeuille d'actions de la Bourse de Casablanca et leur poids.
actions = ["Attijariwafa Bank", "Maroc Telecom", "BCP", "Cosumar", "Marsa Maroc"]
poids = np.array([0.30, 0.20, 0.25, 0.15, 0.10])

rendements_moyens = np.array([0.00042, 0.00024, 0.00040, 0.00028, 0.00048])
volatilites = np.array([0.0125, 0.0088, 0.0119, 0.0100, 0.0156])

valeur_portefeuille = 1000000.0   # Valeur totale du portefeuille (dirhams)
niveau_confiance = 0.95           # Niveau de confiance de la VaR (95%)
horizon_jours = 1                 # Horizon de calcul (1 jour)


def construire_matrice_covariance():
    """
    Construit la matrice de covariance des rendements journaliers des actions.

    Elle combine les correlations entre actions et leurs volatilites. C'est
    l'element cle qui capture le fait que la diversification reduit le risque
    global du portefeuille.
    """
    correlations = np.array([
        [1.00, 0.30, 0.65, 0.25, 0.35],
        [0.30, 1.00, 0.28, 0.22, 0.18],
        [0.65, 0.28, 1.00, 0.26, 0.33],
        [0.25, 0.22, 0.26, 1.00, 0.20],
        [0.35, 0.18, 0.33, 0.20, 1.00],
    ])

    n = len(actions)
    covariance = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            covariance[i][j] = correlations[i][j] * volatilites[i] * volatilites[j]

    return covariance


def var_parametrique(covariance):
    """
    Calcule la VaR par la methode parametrique (dite variance-covariance).

    On suppose que les rendements suivent une loi normale. La VaR se deduit
    alors directement de la volatilite du portefeuille et du niveau de
    confiance, via le quantile de la loi normale. C'est la methode la plus
    rapide, mais elle suppose une distribution normale des rendements.
    """
    rendement_portefeuille = np.sum(poids * rendements_moyens)
    variance_portefeuille = np.dot(poids, np.dot(covariance, poids))
    volatilite_portefeuille = np.sqrt(variance_portefeuille)

    z = norm.ppf(1 - niveau_confiance)
    var_relative = -(rendement_portefeuille + z * volatilite_portefeuille)
    var_montant = var_relative * valeur_portefeuille * np.sqrt(horizon_jours)

    return var_montant, volatilite_portefeuille


def var_historique(rendements_simules):
    """
    Calcule la VaR par la methode historique.

    On classe les rendements passes (ou simules) du portefeuille du pire au
    meilleur, et on lit la perte correspondant au niveau de confiance choisi.
    Cette methode ne suppose aucune forme de distribution : elle laisse les
    donnees parler d'elles-memes.
    """
    seuil = np.percentile(rendements_simules, (1 - niveau_confiance) * 100)
    var_montant = -seuil * valeur_portefeuille
    return var_montant


def var_monte_carlo(covariance, n_simulations=100000):
    """
    Calcule la VaR par simulation de Monte Carlo.

    On simule un grand nombre de scenarios de rendements du portefeuille en
    respectant les correlations entre actions, puis on en deduit la VaR comme
    pour la methode historique. C'est l'approche la plus flexible, capable de
    gerer des portefeuilles complexes.
    """
    rendements_actions = np.random.multivariate_normal(
        rendements_moyens, covariance, n_simulations
    )
    rendements_portefeuille = rendements_actions.dot(poids)

    seuil = np.percentile(rendements_portefeuille, (1 - niveau_confiance) * 100)
    var_montant = -seuil * valeur_portefeuille

    return var_montant, rendements_portefeuille


def calculer_cvar(rendements_simules):
    """
    Calcule la CVaR (Conditional VaR), aussi appelee Expected Shortfall.

    La VaR indique un seuil de perte, mais ne dit rien sur l'ampleur des pertes
    lorsqu'on depasse ce seuil. La CVaR comble ce manque : elle mesure la perte
    moyenne dans les pires scenarios, au-dela de la VaR. C'est la mesure exigee
    par la reglementation de Bale la plus recente, car elle capture mieux les
    risques extremes.
    """
    seuil = np.percentile(rendements_simules, (1 - niveau_confiance) * 100)
    pertes_extremes = rendements_simules[rendements_simules <= seuil]
    cvar_montant = -np.mean(pertes_extremes) * valeur_portefeuille
    return cvar_montant


def afficher_resultats(covariance):
    """Calcule et compare la VaR par les trois methodes, plus la CVaR."""
    print("Portefeuille analyse")
    tableau = pd.DataFrame({
        "Action": actions,
        "Poids": [f"{p*100:.0f}%" for p in poids],
    })
    print(tableau.to_string(index=False))
    print(f"Valeur du portefeuille : {valeur_portefeuille:,.0f} dirhams")
    print(f"Niveau de confiance : {niveau_confiance*100:.0f}%")
    print(f"Horizon : {horizon_jours} jour")
    print()

    var_param, vol = var_parametrique(covariance)
    var_mc, rendements_simules = var_monte_carlo(covariance)
    var_hist = var_historique(rendements_simules)
    cvar = calculer_cvar(rendements_simules)

    print("Value at Risk par methode")
    print(f"VaR parametrique : {var_param:,.0f} dirhams")
    print(f"VaR historique : {var_hist:,.0f} dirhams")
    print(f"VaR Monte Carlo : {var_mc:,.0f} dirhams")
    print()
    print(f"CVaR (perte moyenne au-dela de la VaR) : {cvar:,.0f} dirhams")
    print()
    print(f"Interpretation : avec {niveau_confiance*100:.0f}% de confiance, la perte")
    print(f"sur 1 jour ne devrait pas depasser environ {var_mc:,.0f} dirhams.")
    print()

    return rendements_simules, var_mc, cvar


def tracer_distribution(rendements_simules, var_mc, cvar):
    """Trace la distribution des pertes avec les seuils de VaR et CVaR."""
    pertes_dirhams = rendements_simules * valeur_portefeuille

    plt.figure(figsize=(11, 6))
    plt.hist(pertes_dirhams, bins=120, color="#4C72B0", alpha=0.7,
             edgecolor="none")

    plt.axvline(-var_mc, color="#C44E52", linestyle="--", linewidth=2,
                label=f"VaR 95% : {var_mc:,.0f} dh")
    plt.axvline(-cvar, color="darkred", linestyle="--", linewidth=2,
                label=f"CVaR 95% : {cvar:,.0f} dh")

    plt.xlabel("Gain / Perte sur le portefeuille (dirhams)")
    plt.ylabel("Frequence des scenarios")
    plt.title("Distribution des pertes possibles du portefeuille (1 jour)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("var_resultats.png", dpi=120)
    print("Graphique enregistre : var_resultats.png")


if __name__ == "__main__":
    np.random.seed(42)
    covariance = construire_matrice_covariance()
    rendements_simules, var_mc, cvar = afficher_resultats(covariance)
    tracer_distribution(rendements_simules, var_mc, cvar)
