import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

def resize_with_padding(image, target_size=(256, 256)):
    # Obtention de la taille de l'image d'origine
    width, height = image.size

    # Calcul du ratio de redimensionnement
    ratio = min(target_size[0] / width, target_size[1] / height)

    # Calcul de la taille de l'image redimensionnée
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Redimensionnement de l'image avec LANCZOS à la place d'ANTIALIAS dépréciée depuis Pillow==10.0.0
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Calcul de la couleur des bords
    border_color = image.getpixel((0, 0))

    # Création d'une nouvelle image avec la taille cible et remplissage des pixels avec la couleur des bords
    padded_image = Image.new('RGB', target_size, border_color)
    offset = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)
    padded_image.paste(resized_image, offset)

    return padded_image


def resize_and_save_images(image_paths, target_size=(256, 256), output_folder="image_redimensionnees"):
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, image_path in enumerate(image_paths):
        # Charger l'image
        image = Image.open(image_path)

        # Redimensionner l'image avec le padding
        resized_image = resize_with_padding(image, target_size)

        # Obtenir le nom de fichier sans extension
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Définir le chemin de sortie dans le nouveau dossier
        output_path = os.path.join(output_folder, f"{filename}.jpg")

        # Enregistrer l'image redimensionnée
        resized_image.save(output_path)

        if (i + 1) % 100 == 0 or i == len(image_paths) - 1:
            print(f"{i + 1} images enregistrées")
  

# Tracer le graphique de la variance expliquée cumulée en fonction du nombre de composants
def variance_needed_plot(cumulative_var, n_components):
    plt.plot(cumulative_var)
    plt.axvline(x=n_components, color='r', linestyle='--')
    plt.axhline(y=0.99, color='g', linestyle='--')
    plt.title("Accumulation du pourcentage de l'explication la variance \nen fonction du nombre de composants", size=12)
    plt.xlabel('Nombre de composants', size=11)
    plt.text(n_components-20, 0.3, f'n_components={n_components}', rotation=90, size=11, color='r')
    plt.text(1, 1, f'99% de la variance expliquée', rotation=0, size=11, color='g')
    plt.ylabel('Variance expliquée cumulée', size=11)
    plt.show()          

def perform_dimensionality_reduction(features, use_pca=True, perplexity=30, n_iter=2000, n_clusters=7, random_state=None):
    if use_pca:
        pipeline = Pipeline([
            ('reducDimension', PCA()),
        ])

        # Entraîner le pipeline
        pipeline.fit(features)
        
        # Calculer la variance expliquée cumulée
        explained_variance = pipeline.named_steps['reducDimension'].explained_variance_ratio_
        cumulative_var = np.cumsum(explained_variance)
        
        # Trouver le nombre de dimensions nécessaires pour atteindre 0.99 de la variance expliquée
        n_components = np.argmax(cumulative_var >= 0.99)
        variance_needed_plot(cumulative_var=cumulative_var, n_components=n_components)
    else:
        pass

    class TSNETransformer(BaseEstimator, TransformerMixin):
        def __init__(self, n_components=2, perplexity=30, n_iter=2000, random_state=None):
            self.n_components = n_components
            self.perplexity = perplexity
            self.n_iter = n_iter
            self.random_state = random_state
            self.tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=random_state)

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return self.tsne.fit_transform(X)

    # Définir la fonction de transformation qui convertit la sortie de TSNE en un tableau de caractéristiques
    def tsne_features(tsne_output):
        return tsne_output[:, :2]
    
    if use_pca: 
        # Créer le pipeline
        pipeline = Pipeline([
            # Réduction des dimensions pour ne garder que 99% de l'explication de la variance
            ('reducDimension', PCA(n_components=n_components)),
            # Réduction TSNE 
            ('tsne', TSNETransformer(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)),
            # Transformation pour être compatible avec le reste du pipeline
            ('tsne_features', FunctionTransformer(tsne_features)),
            # Création des 7 clusters
            ('cluster', KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')),
        ])
    else: 
        pipeline = Pipeline([
            # Réduction TSNE 
            ('tsne', TSNETransformer(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)),
            # Transformation pour être compatible avec le reste du pipeline
            ('tsne_features', FunctionTransformer(tsne_features)),
            # Création des 7 clusters
            ('cluster', KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')),
        ])

    # Entraîner le pipeline
    pipeline.fit(features)

    # Transformation du pipeline
    X_embedded = pipeline.transform(features)

    # Prédiction des labels des clusters
    pred_label = pipeline.predict(features)

    return X_embedded, pred_label, pipeline


def perform_dimensionality_reduction_Kfold(features, use_pca=True, perplexity=30, n_iter=2000, n_clusters=7, random_state=None, n_splits=3):
    if use_pca:
        pipeline = Pipeline([
            ('reducDimension', PCA()),
        ])

        # Entraîner le pipeline
        pipeline.fit(features)
        
        # Calculer la variance expliquée cumulée
        explained_variance = pipeline.named_steps['reducDimension'].explained_variance_ratio_
        cumulative_var = np.cumsum(explained_variance)
        
        # Trouver le nombre de dimensions nécessaires pour atteindre 0.99 de la variance expliquée
        n_components = np.argmax(cumulative_var >= 0.99)
        variance_needed_plot(cumulative_var=cumulative_var, n_components=n_components)
    else:
        pass

    class TSNETransformer(BaseEstimator, TransformerMixin):
        def __init__(self, n_components=2, perplexity=30, n_iter=2000, random_state=None):
            self.n_components = n_components
            self.perplexity = perplexity
            self.n_iter = n_iter
            self.random_state = random_state
            self.tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=random_state)

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return self.tsne.fit_transform(X)

    # Définir la fonction de transformation qui convertit la sortie de TSNE en un tableau de caractéristiques
    def tsne_features(tsne_output):
        return tsne_output[:, :2]
    
    if use_pca: 
        # Créer le pipeline
        pipeline = Pipeline([
            # Réduction des dimensions pour ne garder que 99% de l'explication de la variance
            ('reducDimension', PCA(n_components=0.99)),
            # Réduction TSNE 
            ('tsne', TSNETransformer(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)),
            # Transformation pour être compatible avec le reste du pipeline
            ('tsne_features', FunctionTransformer(tsne_features)),
            # Création des 7 clusters
            ('cluster', KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')),
        ])
    else: 
        pipeline = Pipeline([
            # Réduction TSNE 
            ('tsne', TSNETransformer(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)),
            # Transformation pour être compatible avec le reste du pipeline
            ('tsne_features', FunctionTransformer(tsne_features)),
            # Création des 7 clusters
            ('cluster', KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')),
        ])

    # Entraîner le pipeline avec K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    for train_index, test_index in kf.split(features):
        # Séparer les données en ensembles d'entraînement et de test
        X_train, X_test = features[train_index], features[test_index]

        # Entraîner le pipeline sur l'ensemble d'entraînement
        pipeline.fit(X_train)

        # Évaluer le pipeline sur l'ensemble de test
        score = pipeline.score(X_test)

        scores.append(score)

    # Calculez la moyenne des scores de validation croisée
    mean_score = np.mean(scores)

    # Transformation du pipeline
    X_embedded = pipeline.transform(features)

    # Prédiction des labels des clusters
    pred_label = pipeline.predict(features)

    return X_embedded, pred_label, pipeline