# Third parties
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn.metrics import homogeneity_score, v_measure_score, completeness_score
from sklearn.metrics import confusion_matrix, adjusted_rand_score, classification_report

# Internal
from IPython.display import display
import os

load_dotenv()
RESULTS_PATH =f'{os.getenv("ABSOLUTE_PATH")}model_results'
 
# Modification des labels prédits pour les rapprocher le plus possible des vrais labels
def conf_mat_transform(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    corresp = np.argmax(conf_mat, axis=0)
    y_pred_transformed = corresp[y_pred]
    return y_pred_transformed

# Créer le barplot avec seaborn
def premiere_evaluation_graphique(pred_label, palette, le):
    # Convertir les prédictions en libellés
    df_pred = pd.DataFrame({'pred_label': le.inverse_transform(pred_label)})

    # Groupby sur les prédictions et compter les occurrences
    df_count = df_pred.groupby('pred_label').size().reset_index(name='count')

    # Calcul des pourcentages
    total_count = df_count['count'].sum()
    df_count['Pourcentage'] = (df_count['count'] / total_count) * 100  
    
    # Créer le barplot avec seaborn
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(5,5))
    sns.barplot(y='pred_label', x='count', data=df_count, palette=palette)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.xlabel('Prédiction', fontsize=14, fontweight='bold')
    plt.ylabel('Nombre d\'occurrences', fontsize=14, fontweight='bold')
    plt.title('Occurrences des prédictions', fontsize=16, fontweight='bold')
    
    # Annoter les barres avec les décomptes et les pourcentages
    for index, row in df_count.iterrows():
        ax.text(row['count'] + 1, index, f"{row['count']}\n({row['Pourcentage']:.2f}%)", va='center', fontsize=12, fontweight='bold')
    plt.show()

# fonction d'évaluation d'un modèle présentés sous forme d'un tableau
def eval_metrics_df(true_label, pred_label):
    v_measure = np.round(v_measure_score(true_label, pred_label),4)
    completeness = np.round(completeness_score(true_label, pred_label),4)
    homogeneity = np.round(homogeneity_score(true_label, pred_label),4)
    ARI = np.round(adjusted_rand_score(true_label, pred_label),4)
    metric_df = pd.DataFrame([[v_measure, completeness, homogeneity, ARI]], 
                             columns=['v_measure', 'completeness', 'homogeneity', 'ARI'])
    return metric_df  

# Affichage graphique représentant les données réduites en 2D (CE2)
def visu_fct(X_embedded, true_label, pred_label, my_cmap, norm, cat_list):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot pour les catégories réelles
    ax1 = axes[0]
    scatter1 = ax1.scatter(X_embedded[:, 0], X_embedded[:, 1], c=true_label, cmap=my_cmap, norm=norm)
    ax1.legend(handles=scatter1.legend_elements()[0], labels=cat_list, loc="best", title="Catégorie")
    ax1.set_xlabel('tsne1', fontsize=15, fontweight='bold')
    ax1.set_ylabel('tsne2', fontsize=15, fontweight='bold')
    ax1.set_title('Représentation des descriptions\npar catégories réelles', fontsize=18, pad=5, fontweight='bold')

    # Plot pour les clusters
    ax2 = axes[1]
    scatter2 = ax2.scatter(X_embedded[:, 0], X_embedded[:, 1], c=pred_label, cmap=my_cmap, norm=norm)
    ax2.legend(handles=scatter2.legend_elements()[0], labels=set(pred_label), loc="best", title="Clusters")
    ax2.set_xlabel('tsne1', fontsize=15, fontweight='bold')
    ax2.set_ylabel('tsne2', fontsize=15, fontweight='bold')
    ax2.set_title('Représentation des descriptions\npar clusters', fontsize=18, pad=5, fontweight='bold')
    
    plt.show()

# Création d'un rapport visuel final pour un modèle
def evaluate_model(model_name, X_embedded, pred_label, true_label, le, my_cmap, norm, cat_list, palette):
    # Transformation des pred_label pour les faire matcher avec les true_label 
    pred_label = conf_mat_transform(true_label, pred_label)
    
    # Representation quantitative de chaque cluster
    premiere_evaluation_graphique(pred_label, palette, le)

    # Création d'une ligne de dataframe contenant les quatre métriques d'évaluation 
    # que sont V-measure, la complétude, homogénéité et l'ARI
    results_df = eval_metrics_df(true_label, pred_label)

    # Enregistrement des résultats dans un tableau 
    os.makedirs(RESULTS_PATH, exist_ok=True)    
    results_df.to_csv(f'{RESULTS_PATH}{model_name}_results.csv', index=False)

    # Représentation graphique des données (CE1)
    visu_fct(X_embedded, true_label, pred_label, my_cmap, norm, cat_list)
    
    # Evaluation des résultats
    display(results_df)
    
    # Affichage des métriques d'évaluation precision, recall, f1-score, support
    detailed_report = pd.DataFrame(classification_report(true_label, pred_label, output_dict=True)).transpose().round(2)
    
    # Si l'ARI est supérieur à 0.4, détail supplémentaire à representer
    # dans le rapport 
    if results_df['ARI'].values > 0.0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))

        # Visualisation de la matrice de confusion
        conf_mat = confusion_matrix(true_label, pred_label)
        df_cm = pd.DataFrame(conf_mat, index=[label for label in cat_list], columns=[i for i in "0123456"])
        #sns.set(style="white")
        sns.heatmap(df_cm, annot=True, cmap='YlOrBr', fmt='d', cbar=False, 
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax1)
        ax1.set_title('Matrice de confusion entre les clusters créés\net les catégories des produits réelles')
        ax1.set_xlabel('Clusters')
        ax1.set_ylabel('Catégories')
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)

        # Visualisation des éléments mal triés
        df_pred = pd.DataFrame({'true_label': le.inverse_transform(true_label), 'pred_label': le.inverse_transform(pred_label)})
        df_errors = (df_pred.groupby(['true_label', 'pred_label']).size().reset_index(name='count')[['true_label', 'pred_label', 'count']])
        sns.barplot(data=df_errors, y='true_label', x='count', hue='pred_label', hue_order=cat_list, palette=palette, ax=ax2)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Catégorie prédite')
        ax2.set_xlabel("Nombre de produits")
        ax2.set_ylabel("Catégorie réelle")
        ax2.yaxis.set_label_position("right")
        ax2.set_title("Visualisation sur erreurs de catégorisation:\nNombre de produits prédits en fonction de leur vraie catégorie")

        # Ajouter de l'espace entre les deux graphiques
        plt.subplots_adjust(wspace=0.4)

        # Affichez les deux sous-graphiques côte à côte
        plt.tight_layout()
        plt.show()

        # Affichage du rapport détaillé final
        display(detailed_report)
          
    return pred_label, results_df, detailed_report

