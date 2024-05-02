def continuer():
    while True:
        choix = input("Tapez 1 pour continuer ou 0 pour quitter : ")
        if choix == '1':
            print("Continuation...")
            # Effectuez ici les actions que vous voulez réaliser après avoir appuyé sur 1
            break
        elif choix == '0':
            print("Quitting...")
            return  # Quitte la fonction
        else:
            print("Choix invalide. Veuillez taper 1 pour continuer ou 0 pour quitter.")

# Utilisation de la fonction continuer
print("Début du programme")
continuer()
print("Suite du programme")