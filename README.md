# DS_Master_project_6
This repository contains everything related to project 6 of OpenClassrooms' Data Scientist program.

#create a new repository on the command line
echo "# projet_6" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/a-terrien/projet_6.git
git push -u origin main


Dans ce répertoire sont traités les sujets suivants:

Vscode, Python (version 3.9.17), 


librairies: dans le fichier requirements.txt

Le contenu du répertoire projet_6:

les dossiers: 
- input: avec les images et textes originales et récupérés du contenu d'OpenClassrooms
- output:

les notebook:

## Enoncé condensé (basé sur le vrai énoncé)
Ce sujet traite de la faisabilité d'une catégorisation automatique des nouveaux produits vendus via "Place de marché”, société dans l'e-commerce.
Le but est de créer des modèles non-supervisés puis supervisés de type classification textuelle puis de type classification visuelle. 

### Déroulé du sujet
Le sujet sera sectionné en trois sous-partie:
1. Création de modèles non supervisés se basant sur du texte puis sur des images
2. Création de modèles supervisés se basant sur le meilleur modèle non-supervisé
3. Création d'une API 

### Les étapes à faire
Dans une première partie, il faut faire des modèles non-supervisés passant par les étapes suivantes: 
- Un prétraitement des données texte ou image suivant le cas ;
- Une extraction de features ;
- Une réduction en 2 dimensions, afin de projeter les produits sur un graphique 2D, sous la forme de points dont la couleur correspondra à la catégorie réelle ;
- Analyse du graphique afin d’en déduire ou pas, à l’aide des descriptions ou des images, la faisabilité de regrouper automatiquement des produits de même catégorie ;
- Réalisation d’une mesure pour confirmer ton analyse visuelle, en calculant la similarité entre les catégories réelles et les catégories issues d’une segmentation en clusters.

### Les contraintes à prendre en compte
Dans la première partie où des modèles non-supervisés ont été testés, voici les approches d'extraction texte à utiliser : 
- de type “bag-of-words”, comptage simple de mots et Tf-idf ;
- de type word/sentence embedding classique avec Word2Vec, puis BERT et enfin USE (Universal Sentence Encoder) 

Dans cette même partie, voici les algorithmes d'extraction image à utiliser:
- de type SIFT/ORB/SURF
- de type CNN transfer Learning

Dans la deuxième partie du sujet où au moins un modèle supervisé est testé, il faut utiliser la data augmentation pour optimiser le modèle. 

Dans la dernière partie du sujet, il faut utiliser une API pour ramener les données des 10 premiers produits à base de “champagne” dans un fichier CSV provenant de ce [database](https://rapidapi.com/edamam/api/edamam-food-and-grocery-database) où les champs suivants ont été récupérés: foodId, label, category, foodContentsLabel et enfin image.

### Matériels supplémentaires fournis par OpenClassrooms
Des documents ont été fournis par OpenClassrooms pour gagner du temps dans le traitement du sujet en question. Ils peuvent être retrouvés dans le dossier oc_help. 

### Clause de non-responsabilité
OC admet qu'il n’y avait aucune contrainte de propriété intellectuelle sur les données et les images dans [l'énoncé](https://openclassrooms.com/fr/paths/164/projects/631/assignment).
*[23/04/2024]*. 


### Référentiel d'évaluation

#### Prétraiter des données textes pour obtenir un jeu de données exploitable
- [ ] Nettoyage des champs de texte (suppression de la ponctuation et des mots de liaison, mise en minuscules)
- [ ] Ecriture d'une fonction permettant de “tokeniser” une phrase.
- [ ] Ecriture d'une fonction permettant de “stemmer” une phrase.
- [ ] Ecriture d'une fonction permettant de “lemmatiser” une phrase.
- [ ] Feature engineering de type bag-of-words (bag-of-words standard : comptage de mots, et Tf-idf), avec des étapes de nettoyage supplémentaires : seuil de fréquence des mots, normalisation des mots.
- [ ] Illustration des 5 étapes précédentes sur une phrase test.
- [ ] Mis en oeuvre 3 démarches de word/sentence embedding : Word2Vec, BERT et USE
- [ ] Vérification du respect de la propriété intellectuelle 

#### Prétraiter des données images pour obtenir un jeu de données exploitable
- [ ] Utilisation de librairies spécialisées pour un premier traitement du contraste (ex. : openCV). 
- [ ] Traitement d'images (par exemple passage en gris, filtrage du bruit, égalisation, floutage) sur un ou plusieurs exemples. 
- [ ] Feature engineering de type "bag-of-images" via la génération de descripteurs (algorithmes ORB, ou SIFT, ou SURF). 
- [ ] Feature engineering via un algorithme de Transfer Learning basé sur des réseaux de neurones, comme par exemple CNN. 
- [ ] Vérification du respect de la propriété intellectuelle 

#### Mettre en œuvre des techniques de réduction de dimension
- [ ] Justification de la réduction de dimension
- [ ] Réduction de dimension adaptée à la problématique (ex. : ACP).
- [ ] Justification le choix des valeurs des paramètres dans la méthode de réduction de dimension retenue (ex. : le nombre de dimensions conservées pour l'ACP)

#### Représentation graphique des données à grandes dimensions
- [ ] Technique de réduction de dimension (via LDA, ACP, T-SNE, UMAP ou autre technique)
- [ ] Visualisation graphique des données réduites en 2D (par exemple affichage des 2 composantes du T-SNE)
- [ ] Analyse graphique en 2D

#### Définir la stratégie de collecte de données en recensant les API disponibles, et réaliser la collecte des données répondant à des critères définis via une API (interface de programmation) en prenant en compte les normes RGPD, afin de les exploiter pour l’élaboration d’un modèle.
- [ ] Exposition de la stratégie de collecte de données et recencement des API disponibles pour la mise en oeuvre du projet
- [ ] Création et test d'une requête pour obtenir les données via l’API
- [ ] Récupération des seuls champs strictement nécessaires. Ici, il s'agit des champs: foodId, label, category, foodContentsLabel et image. 
- [ ] Filtrage de la data pour ne récupérer que les lignes correspondant à l’ingrédient (“ingr”) champagne
- [ ] Stockage des données collectées via l’API dans un fichier utilisable (ex. : fichier CSV ou pickle).
- [ ] Vérification du respect des normes RGPD dans la collecte et du stockage des données
  - [ ] Présentation des 5 grands principes du RGPD 
  - [ ] Vérification de l'utilisation uniquement des données nécessaire pour traiter le sujet

#### Définir la stratégie d’élaboration d’un modèle d'apprentissage profond, concevoir ou ré-utiliser des modèles pré-entraînés (transfer learning) et entraîner des modèles afin de réaliser une analyse prédictive.
- [ ] Exposition de la stratégie d’élaboration d’un modèle pour répondre à un besoin métier (par exemple : choix de conception d’un modèle ré-utilisation de modèles pré-entraînés).
- [ ] Identification de la ou les cibles. 
- [ ] Séparation du jeu de données en jeu d’entraînement, jeu de validation et jeu de test. 
- [ ] Vérification qu'il n'y a pas de fuite d’information entre les deux jeux de données (entraînement, validation et test). 
- [ ] Entrainement sur plusieurs modèles d’apprentissage profond (par exemple à l’aide de la librairie Tensorflow / Keras) en partant du plus simple vers les plus complexes. 
- [ ] Utilisation de modèles à partir de modèles pré-entraînés (technique de Transfer Learning)

#### Évaluer la performance des modèles d’apprentissage profond selon différents critères (scores, temps d'entraînement, etc.) afin de choisir le modèle le plus performant pour la problématique métier.
- [ ] Choix d'une métrique adaptée à la problématique métier, et sert à évaluer la performance des modèles 
- [ ] Justification de la métrique d’évaluation 
- [ ] Evaluation de la performance d’un modèle de référence et comparaison pour évaluer la performance des modèles plus complexes 
- [ ] Calcul, hormis la métrique choisie, d'au moins un autre indicateur pour comparer les modèles (par exemple : le temps nécessaire pour l’entraînement du modèle) 
- [ ] Optimisation d'au moins un des hyperparamètres du modèle choisi (par exemple : le choix de la fonction Loss, le Batch Size, le nombre d'Epochs) 
- [ ] Synthèse comparative des différents modèles, par exemple sous forme de tableau. 

### Utiliser des techniques d’augmentation des données afin d'améliorer la performance des modèles.
- [ ] Techniques d’augmentation des données (ex. pour des images : rotation, changement d’échelle, ajout de bruit…). 
- [ ] Synthèse comparative des améliorations de performance grâce aux différentes techniques d'augmentation de données utilisées (maîtrise de l’overfitting, meilleur score).