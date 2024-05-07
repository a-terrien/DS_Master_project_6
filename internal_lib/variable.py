import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.preprocessing import LabelEncoder

# Création des variables globales
# Déclaration des constantes globales
ABSOLUTE_PATH = '/Users/Audrey/Documents/github_repositories/DS_Master_project_6/'
PATH_input: str = f'{ABSOLUTE_PATH}input/'
PATH_original_images: str = f'{PATH_input}Images/'
PATH_resized_images: str = f'{PATH_input}image_redimensionnee/'
PATH_output: str = f'{ABSOLUTE_PATH}output/'
RANDOM_STATE = 42

# Création des catégories
#PATH_input = '/Users/audreyterrien/Documents/github_repositories/DS_Master_project_6/input/'
df = pd.read_csv(PATH_input+'flipkart_com-ecommerce_sample_1050.csv')    
df['CATEGORY'] = df['product_category_tree'].map(lambda x: x.split("[\"")[1].split(" >>", 1)[0])

# Liste des catégories
le = LabelEncoder()
cat_list = list(np.unique(df.CATEGORY))

# Création du label qui sera utilisé lors de la comparaison ARI
true_label = le.fit_transform(df['CATEGORY'])

# Nombre de clusters
n_clusters = len(cat_list)

# Définir l'ordre des catégories et leurs couleurs respectives
palette = {'Baby Care': 'violet', 
           'Beauty and Personal Care': 'blue', 
           'Computers': 'gray', 
           'Home Decor & Festive Needs': 'green',
           'Home Furnishing': 'orange', 
           'Kitchen & Dining': 'red', 
           'Watches': 'brown'}
          
my_colors = ['violet', 'blue',  'gray', 'green', 'orange', 'red','brown']
my_cmap = ListedColormap(my_colors)

# Fixer les couleurs pour chaque catégorie
boundaries = np.arange(len(cat_list) + 1)
norm = BoundaryNorm(boundaries, my_cmap.N)


def main() -> None:
    PATH_input
    PATH_original_images
    PATH_resized_images
    PATH_output
    norm
    boundaries
    my_cmap
    my_colors
    palette
    n_clusters
    true_label
    cat_list
    le
    df

if __name__ == '__main__':
    main()
