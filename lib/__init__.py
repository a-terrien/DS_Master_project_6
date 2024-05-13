# Importation des packages third parties
from dotenv import load_dotenv

# Importation des packages python interne 
#import os
#import sys
# Importation des variables et des modules personnels
load_dotenv()
from .module_eda import *
from .module_preprocess import *
from .module_visual_report import *
from .module_text_eda import *
from .module_visual_results import *

# Format basé sur les sources suivantes:
# PEP 8 – Style Guide for Python Code (https://peps.python.org/pep-0008/)
# Les Modules(https://docs.python.org/3/tutorial/modules.html#packages)
#print('\n\n\n')
#print(sys.path) 