# Guide d'utilisateur
Afin d'employer la solution produite, il faut appliquer les étapes suivantes:
1. Rouler le script "problematiqueDepart.py" dans le sous-dossier "problematique" (les trois classificateurs s'exécutent successivement).

# Structure du dépôt
Le dépôt suit la même structure que le code de départ. 
En fait, seuls quelques fichiers ont été ajoutés en plus des modifications. 
Le premier d'entre eux est preprocessing.py, contenant les paramètres discriminatoires choisis.
Les méthodes affichant un commentaire "legacy" sont des traces de notre démarche, soit des paramètres effacés car il étaient soit redondant ou alors trop peu discriminants.
Le deuxième est comparator.py, soit un utilitaire permettant de déterminer l'aptitude discriminatoire d'un paramètre. 
Pour l'utiliser, il faut simplement remplacer les valeurs dans le dictionnaire identifié par l'argument "methods" par le nom du paramètre et la fonction python employée (voir exemple) et ensuite appeler le script sans le moindre argument.
# Structure du code
Tel qu'expliqué précédemment, le code est fortement basé sur le code de départ fourni et implémente donc des fonctions établies précédemment dans la solution implémentée (avec quelques modifications de qualité de vie).