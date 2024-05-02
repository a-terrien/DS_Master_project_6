from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder 
from .module_categorization import palette, my_cmap, norm, cat_list
from sklearn.metrics import (homogeneity_score, v_measure_score, completeness_score, 
                             confusion_matrix, adjusted_rand_score, classification_report)
# Liste des catégories
le = LabelEncoder()

# Print en gras le texte
def bold_print(texte):
    bold_text = f"\033[1m{texte}\033[0m" 
    print(bold_text)

# Fonction créant un seuil de fréquence de mots (CE5)
def freq_threshold(X, threshold):
    """
    Supprime les mots dont la fréquence est inférieure à un certain seuil.
    """
    # Calcul de la fréquence de chaque mot
    freq = np.ravel(X.sum(axis=0))
    # Indices des mots à garder
    idx = np.where(freq >= threshold)[0]
    return X[:, idx]
  #('threshold', FunctionTransformer(freq_threshold)),

# Tracer le graphique de la variance expliquée cumulée en fonction du nombre de composants
def variance_needed_plot(cumulative_var, n_components):
  plt.plot(cumulative_var)
  plt.axvline(x=n_components, color='r', linestyle='--')
  plt.axhline(y=0.99, color='g', linestyle='--')
  plt.title("Accumulation du pourcentage de l'explication la variance \nen fonction du nombre de composants", size=12)
  plt.xlabel('Nombre de composants', size=11)
  plt.text(n_components-40, 0.3, f'n_components={n_components}', rotation=90, size=11, color='r')
  plt.text(1, 1, f'99% de la variance expliquée', rotation=0, size=11, color='g')
  plt.ylabel('Variance expliquée cumulée', size=11)
  plt.show()

# Créer le barplot avec seaborn
def premiere_evaluation_graphique(pred_label):
  # Convertir les prédictions en libellés
  df_pred = pd.DataFrame({'pred_label': le.inverse_transform(pred_label)})
  
  # Groupby sur les prédictions et compter les occurrences
  df_count = df_pred.groupby('pred_label').size().reset_index(name='Décompte')
  display(df_count)

  # Créer le barplot avec seaborn
  #sns.set_style('whitegrid')
  fig, ax = plt.subplots(figsize=(10,6))
  sns.barplot(y='pred_label', x='Décompte', data=df_count, palette=palette)
  plt.xticks(rotation=45, ha='right', fontsize=12)
  plt.xlabel('Prédiction', fontsize=14, fontweight='bold')
  plt.ylabel('Nombre d\'occurrences', fontsize=14, fontweight='bold')
  plt.title('Occurrences des prédictions', fontsize=16, fontweight='bold')
  plt.show()

# Affichage graphique représentant les données réduites en 2D (CE2)
def visu_fct(X_reduction_model, true_label, pred_label):
    fig = plt.figure(figsize=(15,6))
    
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_reduction_model[:,0],
                         X_reduction_model[:,1], 
                         c=true_label, 
                         cmap=my_cmap,
                         norm=norm)
    ax.legend(handles=scatter.legend_elements()[0], 
              labels=cat_list, 
              loc="best", 
              title="Catégorie")
    plt.xlabel('tsne1', fontsize = 15, fontweight = 'bold')
    plt.ylabel('tsne2', fontsize = 15, fontweight = 'bold')
    plt.title('Représentation des descriptions\npar catégories réelles',
              fontsize=18, 
              pad=5, 
              fontweight='bold')
    
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_reduction_model[:,0],
                         X_reduction_model[:,1], 
                         c=pred_label, 
                         cmap=my_cmap,
                         norm=norm)
    ax.legend(handles=scatter.legend_elements()[0], 
              labels=set(pred_label), 
              loc="best", 
              title="Clusters")
    plt.xlabel('tsne1', fontsize = 15, fontweight = 'bold')
    plt.ylabel('tsne2', fontsize = 15, fontweight = 'bold')
    plt.title('Représentation des descriptions\npar clusters', 
              fontsize=18, 
              pad=5, 
              fontweight='bold')
    
    plt.show()

# fonction d'évaluation d'un modèle présentés sous forme d'un tableau
def eval_metrics_df(true_label, pred_label):
    v_measure = np.round(v_measure_score(true_label, pred_label),4)
    completeness = np.round(completeness_score(true_label, pred_label),4)
    homogeneity = np.round(homogeneity_score(true_label, pred_label),4)
    ARI = np.round(adjusted_rand_score(true_label, pred_label),4)
    metric_df = pd.DataFrame([[v_measure, completeness, homogeneity, ARI]], 
                             columns=['v_measure', 
                                      'completeness', 
                                      'homogeneity', 
                                      'ARI'])
    return metric_df

# Modifcation des labels prédits pour les rapprocher le plus possible des vrais labels
def conf_mat_transform(y_true, y_pred) :
    conf_mat = confusion_matrix(y_true,y_pred)
    corresp = np.argmax(conf_mat, axis=0)
    print ("Correspondance des clusters : ", corresp)
    labels = pd.Series(y_true, name="y_true").to_frame()
    labels['y_pred'] = y_pred
    labels['y_pred_transform'] = labels['y_pred'].apply(lambda x : corresp[x]) 
    return labels['y_pred_transform']


# Visualiser de la matrice de confusion sous forme graphique
def confusion_matrix_plot(true_label, pred_label):
  conf_mat = confusion_matrix(true_label, pred_label)
  df_cm = pd.DataFrame(conf_mat, index=[label for label in cat_list],
                      columns=[i for i in "0123456"])

  #sns.set(style="white")
  plt.subplots(figsize=(9, 6))
  sns.heatmap(df_cm, annot=True, cmap='YlOrBr', fmt='d', cbar=False, 
              square=True, linewidths=.5, cbar_kws={"shrink": .5})
  plt.title('Matrice de confusion entre les clusters créés et les catégories des produits réelles')
  plt.xlabel('Clusters')
  plt.ylabel('Catégories')
  plt.xticks(rotation=45)
  plt.yticks(rotation=0)
  plt.tight_layout()
  plt.show()

# Calculer et créer le rapport de classification
def classification_report_df(true_label, pred_label):
  report = classification_report(true_label,pred_label, output_dict=True)
  return pd.DataFrame(report).transpose().round(2)

def error_plot(true_label, pred_label):
  # Créer un dataframe pour les prédictions
  df_pred = pd.DataFrame({'true_label': le.inverse_transform(true_label), 'pred_label': le.inverse_transform(pred_label)})
  df_errors = (
    df_pred
    .groupby(['true_label', 'pred_label'])
    .size()
    .reset_index(name='Décompte')
    [['true_label', 'pred_label', 'Décompte']])
  display(df_errors)
  
  # Création du graphique à barres
  sns.barplot(data=df_errors, y='true_label', x='Décompte', hue='pred_label', hue_order=cat_list, palette=palette)
  # Positionner la boîte de la légende en dehors du graphe
  plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Catégorie prédite')
  # Affichage du graphe
  plt.xlabel("Nombre de produits")
  plt.ylabel("Catégorie réelle")
  plt.title("Visualisation sur erreurs de catégorisation:\nNombre de produits prédits en fonction de leur vrai catégorie")
  plt.show()