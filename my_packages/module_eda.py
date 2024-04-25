
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.patheffects as path_effects
import seaborn as sns 
from wordcloud import WordCloud
import cv2
from PIL import Image
#from dhash import dhash_int

# Création d'une table qui contient les informations générales pour chacune de ses colonnes
def info_table(table):
    """
    Création d'une table qui contient les informations générales pour chacune de ses colonnes
    """
    listName = list(table.columns)
    listType = list(table.dtypes)
    listNotNa = list(table.count())

    listUnique = list(table.nunique())
    listMissing = list(round(table.isna().mean()*100, 2))
    listDuplicates = list([round(table[col].duplicated(
        keep=False).mean()*100, 2) for col in table.columns])

    listDefinition = list(table.columns)
    listColonne = (listDefinition, listType, listNotNa,
                   listUnique, listMissing, listDuplicates)
    df = pd.DataFrame(listColonne, index=(['Description', 'Type du format',
                                           "Nombre de valeurs totale",
                                           'Nombre de valeurs uniques', 
                                           'Pourcentage de valeurs manquantes', 
                                           'Pourcentage de duplicats'
                                           ]), columns=listName).T
    return df


# Création d'une table où le pourcentage de valeurs manquantes de chaque champ est calculé
def missing(df):
    column_name = df.columns.tolist() 
    perc_missing = round(df.isna().mean()*100,3).tolist()
    nb_missing = df.isna().sum()
    df = (pd.DataFrame({'Nom de la colonne' : column_name,
                       'Pourcentage de valeurs manquantes' : perc_missing,
                       'Nombre de valeurs manquantes' : nb_missing})
          .sort_values(by= 'Nombre de valeurs manquantes', ascending = False, inplace = False))
    return df.reset_index(drop=True)


# Ajout de la médiane dans un graphe
def add_median_labels(ax, precision='.1f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{precision}}', ha='center', va='center',
                       fontweight='bold', color='white', fontsize = 15)
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])


## Création d'une fonction graphique pour les variables catégorielles
colors = ["#fff9d2ff","#2e3329ff","#3b6959ff","#beb3a0ff","#bf9000ff","#01c9b6ff","#78909cff"]
def categorical_graph_full(df, feature, titre):
    def autopct(pct): # only show the label when it's > 10%
        return ('%.1f' % pct) if pct > 3 else ''
    
    fig, (ax1, ax2) = plt.subplots(nrows= 1, ncols=2, figsize=(15, 6))
    
    df.groupby([feature]).size().sort_values(ascending=False).plot(kind = "pie", \
                                                                   autopct=autopct, \
                                                                   textprops={'fontsize': 12, 'color' : 'black'}, \
                                                                   colors=colors, 
                                                                   ax = ax1)
    
    ax1.axis('equal')
    
    df.groupby([feature]).size().sort_values(ascending=True).plot(kind = "bar", rot=30, color='#3b6959ff',
                                                                  ax = ax2)

    fig.subplots_adjust(top=.90)
    fig.suptitle(titre, color = 'blue', size = 20)
    fig.show()


## Création d'une fonction graphique pour les variables catégorielles
def categorical_topCountGraph(df, feature, titre, limit_number):
    fig, ax = plt.subplots(nrows= 1, ncols=1, figsize=(15, 6))
    df.groupby([feature]).size().sort_values(ascending=False).iloc[:limit_number].sort_values(ascending=True).plot(kind = "barh", ax= ax)
    fig.suptitle(titre, color = 'blue', size = 20)
    fig.show()


# Création d'un graphe de distribution des valeurs entre deux champs catégoriels
def distribution_entre_deux_cat(target, cat2, titre):
    # Création de la table de contingence
    contingency_table = pd.crosstab(target,cat2)

    # Convertir la table en format long
    df_long = pd.melt(contingency_table.reset_index(), id_vars=['target'], value_vars=contingency_table.columns)

    # Création de la figure
    fig = px.bar(df_long, x='target', y='value', color='second_cat', barmode='stack')

    # Configuration du graphique
    fig.update_layout(
        yaxis_title="Count",
        legend_title="Category",
        title=titre,
        xaxis=dict(type='category'),
        height=600, width=800
    )

    # Affichage de la figure
    fig.show()
    

# Représentation univariée de la répartition des valeurs d'un feature numérique - boxplot et displot
def num_graph(df, i, title):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [5, 1]}, figsize=(8,5))

    # Boxplot
    sns.boxplot(data = df 
                  , x = df[i]
                  , color = "blue"
                  , saturation = 0.7
                  , showmeans=True
                  , meanprops={"marker":"o",
                           "markerfacecolor":"white", 
                           "markeredgecolor":"black",
                           "markersize":"10"}
                  , linewidth = 1.0
                  , ax = ax2)

    # Displot
    sns.histplot(df
                  , x=df[i]
                  , kde = True
                  , color = "blue"
                  , ax=ax1).set_ylabel('Densité', fontsize = 14)

    add_median_labels(ax2)
    
    # Déplacement du titre au milieu et en haut + suppression de l'espace entre les subplots
    fig.subplots_adjust(top=.91, hspace=0.0)

    #Ajout d'un titre général
    fig.suptitle(title, color = 'blue', fontsize = 14)
    fig.show()
    
def check_image_format(directory, formats=[".jpg", ".jpeg", ".png"]):
    """
    Check the format of all images in a directory.

    Parameters:
    directory (str): Path to directory containing images.
    formats (list): List of image formats to check for.

    Returns:
    dict: Dictionary containing counts of images in each format.
    """
    image_counts = {format: 0 for format in formats}

    for file in os.listdir(directory):
        if os.path.splitext(file)[1].lower() in formats:
            image_counts[os.path.splitext(file)[1].lower()] += 1

    return image_counts


# Les couleurs à utiliser pour le WordCloud
def custom_color(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = ['#fcf7c9', '#9b9a8b', '#c7adcc']
    return random.choice(colors)


# Visualisation sous la forme d'un nuage de mots des 300 mots les plus courants 
def wordcloud_plot(corpus, stop_words = []) : 
    # Instantation un nouveau nuage de mots
    wordcloud = WordCloud(
        random_state = 42,
        normalize_plurals = False,
        background_color='black', 
        width = 600, 
        height= 300,
        max_words = 300,
        stopwords = stop_words,
        color_func = custom_color)

    # Application du nuage de mots au texte
    wordcloud.generate(corpus)

    # Affichage d'un nuage de mots
    fig, ax = plt.subplots(1,1, figsize = (9,6))
    # Ajout de l'interpolation = bilinear pour adoucir les choses
    plt.imshow(wordcloud, interpolation='bilinear')
    # Suppression des axes
    plt.axis('off')
    

## Création d'une fonction graphique pour les variables catégorielles
colors = ["#fff9d2ff","#2e3329ff","#3b6959ff","#beb3a0ff","#bf9000ff","#01c9b6ff","#78909cff"]
def categorical_graph_full(df, feature, titre):
    def autopct(pct): # only show the label when it's > 10%
        return ('%.1f' % pct) if pct > 3 else ''
    
    fig, (ax1, ax2) = plt.subplots(nrows= 1, ncols=2, figsize=(12, 5))
    
    df.groupby([feature]).size().sort_values(ascending=False).plot(kind = "pie", \
                                                                   autopct=autopct, \
                                                                   textprops={'fontsize': 12, 'color' : 'black'}, \
                                                                   colors=colors, 
                                                                   ax = ax1)
    
    ax1.axis('equal')
    
    df.groupby([feature]).size().sort_values(ascending=True).plot(kind = "bar", rot=30, color='#3b6959ff',
                                                                  ax = ax2)

    fig.subplots_adjust(top=.90)
    fig.suptitle(titre, color = 'blue', size = 20)
    fig.show()



def get_image_info(photo_list, image_folder_path):
    info_list = []
    for photo in photo_list:
        image_path = os.path.join(image_folder_path, photo)
        if os.path.exists(image_path):
            # Get image size in kbytes
            size_kb = round(os.path.getsize(image_path) / 1024)
            # Load the image using OpenCV
            image = cv2.imread(image_path)
            # Get image dimensions
            height, width, depth = image.shape
            # Get total number of pixels
            pixels = height * width
            # Append the image info to the info_list
            info_list.append((photo, image_path, size_kb, height, width, depth, pixels))
    # Create the dataframe
    columns = ['photo_name', 'image_path', 'size_kb', 'height', 'width', 'depth', 'pixel']
    df = pd.DataFrame(info_list, columns=columns)
    return df




def check_image_corruption(image_list):
    corrupted_images = []
    for image_path in image_list:
        try:
            with Image.open(image_path) as img:
                img.load()
                np.array(img)
        except (IOError, SyntaxError) as e:
            corrupted_images.append(image_path)
            continue
    return corrupted_images

def show_tuple_images(images_tuple):
    """
    Affiche les images d'un tuple donné
    """
    fig, axs = plt.subplots(1, len(images_tuple), figsize=(15, 15))
    
    for i, img_path in enumerate(images_tuple):
        img = cv2.imread(img_path)
        axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i].axis('off')
        axs[i].set_title(os.path.basename(images_tuple[i]))
        
    plt.show()

def find_duplicate_images_with_different_targets(df, duplicates):
    """
    Vérifie si les duplicatas ont la même catégorie (target) dans le dataframe df.
    
    Args:
    - df : pandas.DataFrame, contient les informations des images et de leur catégorie.
    - duplicates : list of tuples, liste de tuples de duplicatas retournée par la fonction find_duplicate_images.
    
    Returns:
    - duplicates_with_same_target : list of tuples, liste de tuples de duplicatas ayant la même catégorie (target).
    """
    # Créer un sous-dataframe avec les doublons trouvés
    sub_df = df[df['IMAGE'].isin(duplicates)]

    # Vérifier si les doublons ont le même target
    targets = sub_df.groupby('IMAGE')['CATEGORY'].unique()

    # Renvoyer les doublons qui ont des targets différents
    return targets[targets.apply(lambda x: len(x) > 1)]

def get_unique_categories(df, duplicates):
    unique_categories = set()
    for dup in duplicates:
        img1, img2 = dup
        cat1 = df.loc[df['IMAGE']==img1, 'CATEGORY'].iloc[0]
        cat2 = df.loc[df['IMAGE']==img2, 'CATEGORY'].iloc[0]
        if cat1 == cat2:
            unique_categories.add(cat1) 
        elif cat1 != cat2:
            unique_categories.add(cat1)
            unique_categories.add(cat2)
    return unique_categories


def find_duplicate_images(directory):
    """
    This function finds duplicate images in a given directory by comparing their hashes.
    It returns a list of tuples, where each tuple contains the paths of the duplicate images.
    """
    # Dictionary to store the hashes of the images
    hashes = {}
    # List to store the duplicate images
    duplicates = []
    # Iterate over all the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            # Load the image and compute its hash
            image_path = os.path.join(directory, filename)
            with open(image_path, 'rb') as f:
                image_hash = dhash_int(Image.open(f))
            # Check if the hash is already in the dictionary
            if image_hash in hashes:
                # If the hash is already in the dictionary, the images are duplicates
                duplicates.append((hashes[image_hash], image_path))
            else:
                # If the hash is not in the dictionary, add it with the image path
                hashes[image_hash] = image_path
    return duplicates