## Travail de veille 
La veille stratégique se définit comme un processus continue de surveillance et d'analyse de l'environnement externe d'une personne ou d'une entreprise pour retrouver les tendances émergentes, les découvertes, les occasions, les défis et les boulversements dans le marché ou dans l'industrie. Elle se compose de la collecte d'information, de son suivi, de son analyse et de sa diffusion (et sa sauvegarde) pour prendre des décisions informées pour atteindre ses objectifs avec les meilleurs moyens possibles.

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

### Convolutional Neural Networks (CNN) 
Les réseaux de neurones convolutifs (CNN) sont une catégorie spécifique de réseaux de neurones, conçus pour traiter efficacement les données d'images. Contrairement aux méthodes traditionnelles d'apprentissage supervisé, où les features sont extraites manuellement, les CNN sont capables d'apprendre automatiquement les features pertinentes à partir des données d'images. Dans les CNN, les images sont traitées par un processus d'extracteur de features, qui utilise des opérations de filtrage par convolution pour détecter des motifs dans l'image. Ces motifs sont ensuite normalisés et/ou redimensionnés, puis filtrés à nouveau pour extraire des features de plus en plus complexes. Les valeurs résultantes sont concaténées dans un vecteur, qui est ensuite utilisé comme entrée pour la classification. Ils se composent de deux blocs principaux : le premier bloc, qui agit comme un extracteur de features, et le second bloc, qui est responsable de la classification. Le premier bloc utilise des opérations de convolution pour extraire des features des images, tandis que le second bloc effectue des combinaisons linéaires et des fonctions d'activation pour classifier les images.Ils sont largement utilisés dans la classification d'images en raison de leur capacité à apprendre automatiquement les features pertinents. Leur architecture spécifique leur permet d'extraire des features de différentes complexités, ce qui les rend particulièrement adaptés aux problèmes de vision par ordinateur. [Source: Monasse, P., & Nadjahi, K. (Mis à jour le 08/07/2022). Cours: Classez et segmentez des données visuelles. Chapitre: [Qu'est ce qu'un réseau de neurones convolutif (ou CNN)?](https://openclassrooms.com/fr/courses/4470531-classez-et-segmentez-des-donnees-visuelles/5082166-quest-ce-quun-reseau-de-neurones-convolutif-ou-cnn)]

## API 
"Une API (application programming interface ou « interface de programmation d’application ») est une interface logicielle qui permet de « connecter » un logiciel ou un service à un autre logiciel ou service afin d’échanger des données et des fonctionnalités.

Les API offrent de nombreuses possibilités, comme la portabilité des données, la mise en place de campagnes de courriels publicitaires, des programmes d’affiliation, l’intégration de fonctionnalités d’un site sur un autre ou l’open data. Elles peuvent être gratuites ou payantes." [Source: CNIL, "Interface de programmation d'application (API)"](https://www.cnil.fr/fr/definition/interface-de-programmation-dapplication-api)

### Classification non-supervisée ou clustering
"L'approche non supervisée consiste à explorer des données sans guide. [... et elle] consiste en l'organisation d'individus en groupes homogènes. En gros, on définit des classes que l'on ne connaît pas à l'avance." 

### Classification supervisée
"L'approche supervisée apprend pour prévoir (une variable quantitative, dans le cas d'une régression ; ou une variable qualitative, dans le cas d'une classification).[... et elle] consiste à ranger les individus dans des classes connues."
- [Cours: Réalisez une analyse exploratoire de données. Chapitre: Découvrez les méthodes factorielles et la classification non supervisée](https://openclassrooms.com/fr/courses/4525281-realisez-une-analyse-exploratoire-de-donnees/5291335-decouvrez-les-methodes-factorielles-et-la-classification-non-supervisee). (Mis à jour le 11/12/2023)

## Approche bag-of-words (aka BOW) ou sac-de-mots 
La méthode "sac de mots" (ou bag of words) est une approche simple pour représenter un document texte numériquement. Chaque document est considéré comme un ensemble de mots uniques, sans tenir compte de leur ordre ou de leur contexte. On crée un vocabulaire avec tous les mots uniques, puis on représente chaque document par un vecteur de la taille du vocabulaire, où chaque élément représente la fréquence d'apparition du mot correspondant. Ces vecteurs sont ensuite utilisés pour former une matrice de documents-termes, utilisée dans les algorithmes de traitement de texte.

Les techniques des sacs de mots passent par des techniques de normalisation des mots. En traitement du langage naturel, c'est le processus de mise en forme des mots pour les rendre plus uniformes et comparables. Cela implique souvent de mettre tous les mots en minuscules, de supprimer la ponctuation et les accents, et de remplacer les variantes orthographiques d'un même mot par la même forme.

Il existe plusieurs approches BOW dont: 
- la __technique du sac de mots - comptage de mots__
- la __technique de Tf-Idf__
  
__Sources:__
- Le site web de référence en traitement du langage naturel, "Natural Language Toolkit" (NLTK)
- Le site web de la bibliothèque Python pour le traitement du langage naturel, "spaCy".

## Bag-of-Words (BoW) ou Sac de mot
Le modèle de "bag of words" est une méthode simple de représentation de documents en termes de mots, sans considération de l'ordre ou du contexte dans lequel ils apparaissent. Chaque document est représenté par un vecteur de la taille du vocabulaire, où chaque élément du vecteur correspond à la fréquence d'apparition d'un mot spécifique dans le document. Cette approche permet de traiter les documents comme des ensembles non ordonnés de mots.
Une caractéristique du modèle "bag of words" est son utilisation de la tokenisation pour séparer le texte en mots individuels ou en groupes de mots appelés n-grammes. Les n-grammes, tels que les bigrammes (paires de mots) ou les trigrammes (groupes de trois mots), capturent davantage d'informations sur les relations entre les mots que les mots individuels. Par exemple, en analysant les bigrammes, on peut prendre en compte la probabilité d'apparition d'un mot en fonction des mots précédents.
Le modèle de "bag of words" est une approche largement utilisée dans le traitement automatique du langage naturel (TALN) pour diverses tâches telles que la classification de texte, l'extraction d'informations et la recherche d'informations.
[Source: Chaouche, Y.(Mis à jour le 06/09/2022). Représentez votre corpus en "bag of words". [Dans Analysez vos données textuelles.](https://openclassrooms.com/fr/courses/4470541-analysez-vos-donnees-textuelles/4855001-representez-votre-corpus-en-bag-of-words)]

## Tf-idf (Term-Frequency - Inverse Document Frequency)
TF-IDF évalue l'importance d'un terme dans un document par rapport à une collection de documents. Il combine la fréquence du terme dans le document avec sa rareté dans le corpus. Les mots communs à tous les documents obtiennent un score faible, tandis que les mots rares dans un document mais fréquents dans d'autres obtiennent un score élevé. Ce score est utilisé dans diverses applications telles que la recherche d'informations et l'extraction de mots-clés.

__Fréquence des termes:__ La fréquence des termes (TF) d'un terme ou mot est le nombre de fois où le terme apparaît dans un document par rapport au nombre total de mots dans le document.
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>T</mi>
  <mi>F</mi>
  <mo>=</mo>
  <mfrac>
    <mtext>number of times the term appears in the document</mtext>
    <mtext>total number of terms in the document</mtext>
  </mfrac>
</math>

__Fréquence inverse des documents:__ La fréquence inverse des documents (IDF) d'un terme reflète la proportion de documents dans le corpus qui contiennent le terme. Les mots uniques à un petit pourcentage de documents (par exemple, des termes de jargon technique) reçoivent des valeurs d'importance plus élevées que les mots communs à tous les documents (par exemple, a, le, et).
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>I</mi>
  <mi>D</mi>
  <mi>F</mi>
  <mo>=</mo>
  <mi>l</mi>
  <mi>o</mi>
  <mi>g</mi>
  <mo stretchy="false">(</mo>
  <mfrac>
    <mtext>number of the documents in the corpus</mtext>
    <mtext>number of documents in the corpus contain the term</mtext>
  </mfrac>
  <mo stretchy="false">)</mo>
</math>
Le TF-IDF d'un terme est calculé en multipliant les scores TF et IDF.
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext mathvariant="italic">TF-IDF</mtext>
  <mo>=</mo>
  <mi>T</mi>
  <mi>F</mi>
  <mo>&#x2217;</mo>
  <mi>I</mi>
  <mi>D</mi>
  <mi>F</mi>
</math>

__Source:__
- Stecanella, B. (2019, May 10). [Understanding TF-IDF: A Simple Introduction](https://monkeylearn.com/blog/what-is-tf-idf/) 
- Karabiber, F. (Ph.D. in Computer Engineering, Data Scientist). [TF-IDF — Term Frequency-Inverse Document Frequency](https://www.learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/#:~:text=Term%20Frequency%20%2D%20Inverse%20Document%20Frequency%20(TF%2DIDF)%20is,%2C%20relative%20to%20a%20corpus).

## Approche word/sentence classique

### Word2Vec
Word2vec est un algorithme utilisé pour l'embedding de mots. Il existe deux variantes : CBOW et Skip-Gram. CBOW prédit le mot cible à partir du contexte, tandis que Skip-Gram prédit le contexte à partir du mot cible. Le modèle CBOW utilise une architecture avec une couche d'embedding, une couche GlobalAveragePooling1D et une couche Dense pour prédire le mot cible. La fonction de perte cross-entropy est généralement utilisée pour entraîner le modèle.
Le word2vec embedding capture les propriétés arithmétiques des mots. Par exemple, la propriété "roi - homme + femme = reine" est explorée pour vérifier si la propriété se propage à d'autres mots comme "woman". Des exemples de mots proches obtenus à partir de propriétés arithmétiques sont fournis.
Il capture également les propriétés sémantiques des mots. Par exemple, la similarité entre les verbes à l'infinitif et les verbes au passé est explorée pour montrer comment le modèle capture cette relation.
[Source: Robert, J. (2020, 18 septembre). [Word2vec : NLP & Word Embedding](https://datascientest.com/nlp-word-embedding-word2vec). Data Scientist.]

### Doc2Vec
Doc2Vec, également appelé Paragraph Vector, est une technique populaire en traitement automatique du langage naturel (NLP) qui permet de représenter des documents sous forme de vecteurs. Introduite comme une extension de Word2Vec, qui représente les mots sous forme de vecteurs numériques, Doc2Vec est utilisé pour apprendre des embeddings de documents. Cette approche utilise des réseaux neuronaux pour apprendre des représentations distribuées de documents, permettant ainsi la comparaison de documents, la classification, le clustering et l'analyse de similarité.

Il existe deux variantes principales de l'approche Doc2Vec :
- __Distributed Memory (DM):__ Cette variante considère le contexte dans lequel un document apparaît pour apprendre une représentation vectorielle fixe pour chaque morceau de texte. Elle utilise une architecture de réseau neuronal avec une couche de projection et une couche de sortie pour créer des vecteurs de mots et de documents.
- __Distributed Bag of Words (DBOW):__ Cette version plus simple de l'algorithme Doc2Vec se concentre sur la distribution des mots dans un texte plutôt que sur leur signification. Elle attribue une représentation vectorielle unique à chaque document sans considérer l'ordre des mots.
  
La principale différence entre DM et DBOW réside dans le fait que DM prend en compte à la fois l'ordre des mots et le contexte du document, ce qui le rend plus puissant pour capturer la signification sémantique des documents. En revanche, DBOW est plus rapide à entraîner et utile pour capturer les propriétés distributionnelles des mots dans un corpus.
Doc2Vec offre plusieurs avantages, notamment la capacité à capturer le sens sémantique des documents, à générer des embeddings de documents pour diverses tâches et à gérer les mots non vus en exploitant leur contexte. Il est également extensible et peut être personnalisé en ajustant divers hyperparamètres.
[Source: GeeksforGeeks. (11 juillet 2023). [Doc2Vec in NLP](https://www.geeksforgeeks.org/doc2vec-in-nlp/)]

### Glove 
GloVe, ou Global Vectors for Word Representation, est un algorithme d'apprentissage non supervisé qui génère des représentations vectorielles ou embeddings de mots. Présenté pour la première fois en 2014 par Richard Socher, Christopher D. Manning et Jeffrey Pennington, GloVe utilise les données de co-occurrence statistique des mots dans un corpus donné pour capturer les relations sémantiques entre les mots.

L'idée fondamentale derrière GloVe est de représenter les mots sous forme de vecteurs dans un espace vectoriel continu, où l'angle et la direction des vecteurs correspondent aux connexions sémantiques entre les mots appropriés. Pour ce faire, GloVe construit une matrice de co-occurrence en utilisant des paires de mots, puis optimise les vecteurs de mots pour minimiser la différence entre l'information mutuelle ponctuelle des mots correspondants et le produit scalaire des vecteurs.

Les embeddings GloVe sont des options populaires pour représenter les mots dans les données textuelles et ont trouvé des applications dans diverses tâches de traitement automatique du langage naturel (NLP). Ils peuvent être utilisés pour la classification de texte, la reconnaissance d'entités nommées (NER), la traduction automatique, les systèmes de question-réponse, la similarité et le regroupement de documents, ainsi que dans les tâches d'analogie de mots et de recherche sémantique. [Source: GeeksforGeeks. (03 janvier 2024). [Pre-trained Word embedding using Glove in NLP models](https://www.geeksforgeeks.org/pre-trained-word-embedding-using-glove-in-nlp-models/?ref=header_search)]

### FastText
FastText est une bibliothèque open source développée par Facebook AI Research (FAIR) pour apprendre l'intégration de mots et la classification de mots. Cette bibliothèque permet de créer des algorithmes d'apprentissage supervisé ou non supervisé pour obtenir des représentations vectorielles de mots. Elle prend en charge les modèles CBOW et Skip-gram. FastText est utilisé pour trouver des similitudes sémantiques et pour la classification de texte, comme le filtrage du spam. Il peut former de grands ensembles de données en quelques minutes, offrant une alternative rapide aux modèles basés sur des réseaux neuronaux profonds. Les représentations de mots générées par FastText contiennent des informations sur les sous-mots, ce qui aide le modèle à établir une similarité sémantique entre les mots. FastText utilise également la technique des N-Gram pour entraîner le modèle, ce qui lui permet de capturer la signification des suffixes/préfixes pour les mots donnés dans le corpus. Il peut être utilisé sur des langues morphologiquement riches comme l'espagnol, le français et l'allemand. [Source: GeeksforGeeks, "Fonctionnement et mise en œuvre de FastText", Dernière mise à jour : 26 novembre 2020]

## Approche word/sentence actuelle
### BERT
BERT, acronyme de Bidirectional Encoder Representations from Transformers, est un modèle de deep learning pré-entraîné développé par la filière d'intelligence artificielle de Google (Google AI) et publié en octobre 2018. Conçu pour le traitement automatique du langage naturel (NLP), BERT a suscité un vif intérêt dans la communauté de la data science en raison de ses performances « State-of-the-art », dépassant même les performances humaines dans certaines tâches. Contrairement aux modèles de NLP pré-BERT, qui adoptaient une approche unidirectionnelle pour comprendre le texte, BERT utilise une méthode bi-directionnelle, lui permettant d'avoir une meilleure compréhension du contexte. En utilisant la technique Masked Language Model (MLM), BERT masque aléatoirement des mots dans une phrase puis tente de les prédire, en prenant en compte à la fois le contexte précédent et suivant. Techniquement, BERT est basé sur l'architecture des Transformers, composée d'un encodeur pour lire le texte et d'un décodeur pour faire des prédictions. Avant d'utiliser BERT, une préparation des données est nécessaire, notamment la tokenisation des mots, l'ajout de tokens de début et de fin de phrase, ainsi que l'ajout de marqueurs de position à chaque token. BERT offre plusieurs modèles de différentes tailles, permettant à l'utilisateur de choisir la complexité adaptée à sa tâche. En intégrant BERT à une architecture, il devient possible de réaliser des prédictions dans diverses applications de NLP, telles que la classification de texte selon le sentiment ou la création d'assistants virtuels intelligents. (Source: ["BERT : Un outil de traitement du langage innovant"](https://datascientest.com/bert-un-outil-de-traitement-du-langage-innovant) par Jérémy Robert, publié le 30 septembre 2021 sur Data Scientist)

### USE (Universal Sentence Encoder)
L'Universal Sentence Encoder (USE) de Google est un outil qui encode les textes en vecteurs numériques de haute dimension, permettant ainsi leur utilisation dans diverses tâches de traitement automatique du langage naturel (TALN) telles que la classification de texte, la similarité sémantique et le regroupement. Le USE est disponible en pré-entraîné dans Tensorflow-hub et offre deux variantes : l'une entraînée avec un encodeur Transformer et l'autre avec un réseau d'agrégation profond (Deep Averaging Network, DAN). Ces deux variantes présentent un compromis entre précision et exigences en ressources computationnelles. Le modèle avec un encodeur Transformer offre une précision plus élevée mais nécessite plus de ressources computationnelles, tandis que celui avec un encodage DAN est moins intensif en ressources mais avec une précision légèrement inférieure. L'USE est largement utilisé comme couche d'incorporation (embedding layer) au début des modèles de Deep Learning pour le traitement de texte. (Source: ["Use-cases of Google’s Universal Sentence Encoder in Production"](https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15) par Sambit Mahapatra, publié le 24 janvier 2019 sur Towards Data Science)

## Data Augmentation
La data augmentation consiste à augmenter artificiellement la quantité de données disponibles pour les modèles de Deep Learning en générant de nouveaux points de données à partir des données existantes, en apportant des modifications mineures ou en utilisant d'autres modèles d'apprentissage automatique. Cette méthode permet d'améliorer la diversité et la qualité des ensembles de données d'entraînement, conduisant ainsi à des modèles plus performants. Cependant, elle peut conserver ou même renforcer les biais des données originales et nécessite des ressources importantes pour garantir la qualité des données synthétiques. (Source: [datascientest.com](https://datascientest.com/data-augmentation-tout-savoir))
1 Sep 2023 - Jérémy Robert

## Tokenisation
D'après la définition d'OC dans le cours Analysez vos données textuelles -  Récupérez et explorez le corpus de textes: *["le terme token désigne généralement un mot et/ou un élément de ponctuation. [Ainsi] La phrase "Hello World!" comprend donc 3 tokens. [...] La tokenisation désigne le découpage en mots des différents documents qui constituent votre corpus"](https://openclassrooms.com/fr/courses/4470541-analysez-vos-donnees-textuelles/4470548-recuperez-et-explorez-le-corpus-de-textes)* par Yannis Chaouche (Mis à jour le 06/09/2022), créé par OpenClassrooms, Leading E-Learning Platform. [lu le 25/04/2024]. 
La __tokenisation__ est d'après [OpenClassrooms](https://openclassrooms.com/fr/courses/4470541-analysez-vos-donnees-textuelles/4765461-tokenisation-et-pretraitements-de-texte), le processus de division d'un texte en mots ou en sous-chaînes significatives appelées tokens. Cette technique est souvent utilisée comme étape préliminaire pour le traitement du langage naturel, la recherche d'informations, etc.  

## Part-of-speech (POS) tagging
Le __Part-of-speech tagging (POS tagging)__ est, d'après [Spacy](https://spacy.io/usage/linguistic-features#pos-tagging) le processus de marquage de chaque mot dans un texte avec une étiquette grammaticale correspondant à sa fonction dans la phrase (comme un nom, un verbe, un adjectif, etc.). Cette technique est souvent utilisée pour l'analyse syntaxique et sémantique. 

## Named Entity Recognition (NER)
- le __Named Entity Recognition (NER)__ ou reconnaissance d'entités nommées est d'après [OpenClassrooms](https://openclassrooms.com/fr/courses/4470541-analysez-vos-donnees-textuelles/4801291-reconnaissance-dentites-nommees-ner), le processus de détection et de classification des entités nommées dans un texte, telles que les personnes, les lieux, les organisations, les dates, etc. Cette technique est souvent utilisée pour l'extraction d'informations à partir de textes. 

### Stemmatisation
La __lemmatisation__ est d'après [NLTK](https://www.nltk.org/book/ch03.html), le processus de transformation d'un mot en sa forme canonique (lemme), qui est sa forme de base ou son dictionnaire. Cette technique est souvent utilisée pour normaliser les mots et réduire le nombre de variantes dans le texte.

### Lemmatisation
Le __stemming__ est d'après [NLTK](https://www.nltk.org/book/ch03.html), un processus de réduction d'un mot à sa forme racine en supprimant les suffixes et les préfixes. Cette technique est souvent utilisée pour normaliser les mots et réduire le nombre de variantes dans le texte

## Réduction des dimensions
La réduction de dimension est une technique statistique et basée sur l'apprentissage automatique dans laquelle nous essayons de réduire le nombre de caractéristiques dans notre ensemble de données pour obtenir un ensemble de données avec un nombre optimal de dimensions. L'objectif est d'éviter le surajustement et la malédiction de la dimensionnalité, où un grand nombre de caractéristiques peut nuire aux performances des modèles.
Une méthode courante de réduction de dimension est l'extraction de caractéristiques, dans laquelle nous réduisons le nombre de dimensions en cartographiant un espace de caractéristiques de dimension supérieure vers un espace de caractéristiques de dimension inférieure. L'une des techniques les plus populaires d'extraction de caractéristiques est l'Analyse en Composantes Principales (PCA).
[Source: GeeksforGeeks. (18 juillet 2022). [Reduce Data Dimensionality using PCA – Python](https://www.geeksforgeeks.org/reduce-data-dimentionality-using-pca-python/?ref=header_search)]

### Latent Dirichlet Allocation (LDA)
L'Allocation de Dirichlet Latente (LDA) est un modèle statistique pour découvrir des sujets abstraits, également appelé modélisation de sujets. Il permet la classification non supervisée des documents, similaire à la mise en grappes sur des données numériques, pour trouver des groupes naturels d'éléments (sujets) même lorsque l'on n'est pas sûr de ce que l'on recherche.
L'objectif de LDA est de trouver les sujets auxquels appartient un document, en se basant sur les mots qu'il contient. Il aide à organiser, comprendre, rechercher et résumer automatiquement de grands archives électroniques.
LDA est largement utilisé en traitement du langage naturel, notamment dans la classification de textes, la résumé de texte et la modélisation de sujets.

Kulshrestha, R. (Jul 19, 2019). [A Beginner’s Guide to Latent Dirichlet Allocation(LDA)](https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2). Towards Data Science

### Linear Discriminant Analysis (LDA)
L'Analyse Discriminante Linéaire (LDA), également connue sous le nom d'Analyse Discriminante Normale ou Analyse de Fonction Discriminante, est une technique de réduction de dimensionnalité principalement utilisée dans les problèmes de classification supervisée. Elle facilite la modélisation des distinctions entre les groupes, séparant efficacement deux classes ou plus. LDA opère en projetant des caractéristiques d'un espace de dimension supérieure dans un espace de dimension inférieure. En apprentissage automatique, LDA sert d'algorithme d'apprentissage supervisé spécifiquement conçu pour les tâches de classification, visant à identifier une combinaison linéaire de caractéristiques qui sépare de manière optimale les classes au sein d'un ensemble de données.
[Linear Discriminant Analysis in Machine Learning](https://www.geeksforgeeks.org/ml-linear-discriminant-analysis/). (Last Updated: 20 Mar, 2024). 

### Principal Components Analysis (ou PCA)
L'Analyse en Composantes Principales (ACP) est une méthode largement utilisée en réduction de dimension qui permet de transformer des variables très corrélées en nouvelles variables décorrélées les unes des autres. Son principe est de résumer l'information contenue dans une large base de données en un certain nombre de variables synthétiques appelées "composantes principales". L'objectif est ensuite de projeter ces données sur l'hyperplan le plus proche afin d'obtenir une représentation simple des données tout en conservant au maximum la variabilité entre les individus. 

Voici comment fonctionne l'Analyse en Composantes Principales :
- 1. Centrer et réduire les variables pour atténuer l'effet d'échelle, car elles ne sont pas calculées sur la même base.
- 2. Considérer les données comme une matrice à partir de laquelle des valeurs propres et des vecteurs propres sont calculés. Les vecteurs propres représentent les axes privilégiés selon lesquels une application d'un espace dans lui-même se comporte comme une dilatation.
- 3. Utiliser les valeurs propres pour déterminer le nombre optimal de composantes principales. Par exemple, si le nombre optimal est 2, les données seront représentées sur deux axes.
- 4. Réduire la dimension des données en conservant un maximum d'informations. Par exemple, une réduction de dimension de 9 à 2 axes peut conserver près de 70 % des informations.
- 5. Visualiser l'importance de chaque variable explicative pour chaque axe de représentation à l'aide du cercle des corrélations. Cet outil permet de comprendre quelles variables contribuent le plus à chaque axe.
En résumé, l'ACP permet de réduire la dimension des données tout en conservant un maximum d'informations, ce qui facilite leur interprétation et leur visualisation. 
[Source: datascientest.com, "Qu’est-ce que l’Analyse en Composantes principales ?", Raphael Kassel, 13 Janvier 2021]

### T-SNE
t-SNE, ou t-distributed Stochastic Neighbor Embedding, est une technique de réduction de dimension utilisée en exploration de données de grandes dimensions. Développée en 2008 par Geoffrey Hinton et Laurens van der Maaten, elle propose une approche différente de l'Analyse en Composantes Principales (ACP). Contrairement à l'ACP qui cherche à maximiser la variance des données dans un sous-espace de dimension réduite, t-SNE vise à représenter les données dans un espace de plus petite dimension tout en préservant les distances entre les points.

Le principe de t-SNE consiste à créer une distribution de probabilité pour représenter les similarités entre voisins dans un espace de grande dimension et dans un espace de plus petite dimension. Il se divise en trois étapes :
- 1. Calcul des similarités des points dans l'espace initial en grande dimension en utilisant des distributions gaussiennes centrées sur chaque point. Ces similarités sont normalisées pour chaque point en fonction d'une valeur appelée perplexité, qui contrôle la variance des distributions gaussiennes.
- 2. Création d'un espace de plus petite dimension où les points sont initialement répartis de manière aléatoire. Les similarités des points dans cet espace sont calculées en utilisant une distribution t-Student.
- 3. Comparaison des similarités des points dans les deux espaces à l'aide de la mesure de divergence de Kullback-Leibler (KL), suivie d'une minimisation de cette divergence par descente de gradient pour obtenir les meilleures coordonnées dans l'espace de dimension réduite.
Comparativement à l'ACP, t-SNE est capable de regrouper les données proches et d'éloigner les données dissemblables dans l'espace de dimension réduite, comme illustré par une comparaison des résultats obtenus sur le jeu de données MNIST. [Source: datascientest.com, "Comprendre l’algorithme t-SNE en 3 étapes", Raphael Kassel, 24 Mai 2021]

### Uniform Manifold Approximation and Projection (UMAP)
L'approximation et projection uniforme de variétés (UMAP) est une autre technique de réduction de dimensionnalité et de visualisation. Elle simplifie les données complexes et préserve la structure locale ainsi que les relations entre les points de données. UMAP utilise l'apprentissage de variétés pour comprendre la structure sous-jacente ou la forme des données. Il se concentre sur la capture de relations complexes et non linéaires dans les données qui peuvent ne pas être capturées par PCA ou t-SNE. UMAP est scalable et peut gérer efficacement de grands ensembles de données, ce qui le rend adapté à la visualisation et à l'exploration.

Sung Mo Park. (Publié le 15 juil. 2023) [Easy explanation of the dimension reduction (PCA, t-SNE, and UMAP)](https://www.linkedin.com/pulse/easy-explanation-dimension-reduction-pca-t-sne-umap-sung-mo-park#:~:text=%2D%20PCA%20provides%20a%20global%20view,is%20specifically%20designed%20for%20scalability).