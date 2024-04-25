## Travail de veille 
La veille stratégique se définit comme un processus continue de surveillance et d'analyse de l'environnement externe d'une personne ou d'une entreprise pour retrouver les tendances émergentes, les découvertes, les occasions, les défis et les boulversements dans le marché ou dans l'industrie. Elle se compose de la collecte d'information, de son suivi, de son analyse et de sa diffusion (et sa sauvegarde) pour prendre des décisions informées pour poursuivre ses objectifs le mieux que possible.

__Sources:__
- Article wikipedia: *[Veille stratégique](https://fr.wikipedia.org/wiki/Veille_strat%C3%A9gique), [lu le 25/04/2024]* 
- François Brouard, (Automne 2003) *[La veille stratégique, un outil pour favoriser l'innovation au Canada](https://carleton.ca/profbrouard/wp-content/uploads/2003articleROTveillestrategiqueinnovationBrouardfinal.pdf)*. Organisations & territoires, Volume 12, N°3, Page 54. *[lu le 25/04/2024]*

## Preuve de concept d’une technique récente
*"Une preuve de concept (POC pour Proof of Concept) est une démonstration de faisabilité, c.-⁠à-⁠d. une réalisation expérimentale concrète et préliminaire, courte ou incomplète, illustrant une certaine méthode ou idée afin d’en démontrer ou pas la faisabilité, avec un budget accessible à un chef de projet. Situé très en amont dans le processus de développement d’un produit ou d’un processus nouveau, le POC est une habituellement considéré comme une étape importante sur la voie d’un prototype pleinement fonctionnel."* extrait du guide collaboratif avec le comité Richelieu (Juin 2019) : ["De l'idée à l'industrialisation : réussissez votre preuve de concept"](https://www.economie.gouv.fr/files/files/directions_services/mediateur-des-entreprises/PDF/4_INNOVER_ENSEMBLE/guide-poc.pdf). Chapitre 2 - Qu'est ce qu'une preuve de concept ?, Page 7, publié dans Le Médiateur des entreprises.

### SIFT
SIFT (Scale-Invariant Feature Transform) est un algorithme de traitement d'images utilisé pour extraire des caractéristiques (ou des descripteurs) uniques à partir d'images numériques. Les descripteurs extraits sont utilisés pour effectuer la correspondance de caractéristiques entre les images, pour la reconnaissance d'objets, la reconnaissance de visages, la reconstruction 3D, etc.

Les __hyperparamètres de SIFT__ dépendent de l'implémentation de l'algorithme et peuvent varier. Voici quelques-uns des hyperparamètres les plus courants:

- __nfeatures__ : nombre maximal de caractéristiques à extraire. Par défaut, nfeatures=0, ce qui signifie qu'il n'y a pas de limite.
- __nOctaveLayers__ : nombre de couches d'échelle par octave. Par défaut, nOctaveLayers=3.
- __contrastThreshold__ : seuil de contraste minimal pour détecter une caractéristique. Par défaut, contrastThreshold=0.04.
- __edgeThreshold__ : seuil d'arrêt pour éliminer les caractéristiques à proximité des bords de l'image. Par défaut, edgeThreshold=10.
- __sigma__ : écart-type du flou gaussien appliqué à l'image en entrée pour la construction de l'espace d'échelle. Par défaut, sigma=1.6.

Ces hyperparamètres peuvent être ajustés pour optimiser la performance de l'algorithme en fonction des exigences spécifiques de chaque application.

### ORB
ORB (Oriented FAST and Rotated BRIEF) est un descripteur d'image qui est utilisé pour la détection de points d'intérêt dans des images numériques. Contrairement aux descripteurs SIFT et SURF qui utilisent des convolutions gaussiennes, ORB utilise des filtres binaires pour détecter les points d'intérêt.

Les [hyperparamètres d'ORB](https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html) sont les suivants :

- __nfeatures__ : spécifie le nombre de points d'intérêt à détecter. Par défaut, nfeatures=500.
- __scaleFactor__ : spécifie le facteur d'échelle entre les différentes images de l'échelle pyramidale. Par défaut, scaleFactor=1.2.
- __nlevels__ : spécifie le nombre d'échelles dans l'échelle pyramidale. Par défaut, nlevels=8.
- __edgeThreshold__ : spécifie le seuil de réponse du détecteur de coins pour éliminer les coins situés sur des bords. Par défaut, edgeThreshold=31.
- __firstLevel__ : spécifie le niveau de l'échelle pyramidale à partir duquel commencer la détection de points d'intérêt. Par défaut, firstLevel=0.
- __WTA_K__ : spécifie le nombre de points d'intérêt à comparer dans le calcul du BRIEF. Par défaut, WTA_K=2.
- __scoreType__ : spécifie le type de score pour les points d'intérêt. Par défaut, scoreType=cv.ORB_HARRIS_SCORE.
- __patchSize__ : spécifie la taille de la fenêtre de calcul de l'intensité du pixel pour le calcul de la réponse du détecteur de coins. Par défaut, patchSize=31.
- __fastThreshold__ : spécifie le seuil pour le détecteur de coins FAST. Par défaut, fastThreshold=20.
  
### SURF
SURF (Speeded Up Robust Features) est un algorithme de détection de points clés et de description de l'image, utilisé pour la reconnaissance d'objets et la correspondance d'images. Il a été développé par Herbert Bay, Tinne Tuytelaars et Luc Van Gool en 2006.

Les __[hyperparamètres de SURF](https://docs.opencv.org/4.x/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html)__ sont les suivants :

- __hessianThreshold__ : Seuil pour supprimer les points faibles dans l'image en fonction de l'échelle de l'espace de Harris-Stephens. Les valeurs plus petites suppriment plus de points. Par défaut, hessianThreshold=100.
- __nOctaves__ : Nombre d'octaves à utiliser pour la détection des points. Par défaut, nOctaves=4.
- __nOctaveLayers__ : Nombre de couches par octave. Par défaut, nOctaveLayers=3.
- __extended__ : Si vrai, l'algorithme crée une description 128 bits de chaque point clé. Sinon, 64 bits sont utilisés. Par défaut, extended=False.
- __upright__ : Si vrai, l'algorithme n'utilise pas d'informations sur l'orientation du patch. Par défaut, upright=False.
- __surfMethod__ : Méthode utilisée pour extraire les descripteurs. Il peut s'agir de SURF, SURF_CUDA ou SURF_UPRIGHT. Par défaut, surfMethod=SURF.

Notez que SURF étant breveté, il n'est pas inclus dans certaines distributions d'OpenCV.

### CNN (ou réseau de neurones convolutif) transfer Learning 

## API 

## Classification non-supervisée

## Classification supervisée

## Approche bag-of-words

## Comptage de mots

## Tf-idf

## Approche word/sentence classique

### Word2Vec

### Doc2Vec

### Glove 

### FastText

## Approche word/sentence actuelle
### BERT

### USE (Universal Sentence Encoder)

## Data Augmentation

## Tokenisation
D'après la définition d'OC dans le cours Analysez vos données textuelles -  Récupérez et explorez le corpus de textes: *["le terme token désigne généralement un mot et/ou un élément de ponctuation. [Ainsi] La phrase "Hello World!" comprend donc 3 tokens. [...] La tokenisation désigne le découpage en mots des différents documents qui constituent votre corpus"](https://openclassrooms.com/fr/courses/4470541-analysez-vos-donnees-textuelles/4470548-recuperez-et-explorez-le-corpus-de-textes)* par Yannis Chaouche (Mis à jour le 06/09/2022), créé par OpenClassrooms, Leading E-Learning Platform. [lu le 25/04/2024]. 

## Normalisation des tokens
### Stemmatisation

### Lemmatisation

## Réduction des dimensions

### ACP

### T-SNE

### UMAP

### LDA

## RGPD

