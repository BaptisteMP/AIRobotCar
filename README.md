# AIRobotCar 
Autonomous car race


Voici le fruit de notre projet AI Robot Car, qui se propose de construire à moindre coût une voiture autonome sur un circuit dessiné au sol, en utilisant les réseaux de neurones.
Ce projet s'inscrit dans une perspective de participation aux courses IronCar http://www.ironcar.org/.

Il utilise des ressources en open source développées par Vincent Houlbrèque https://github.com/vinzeebreak/ironcar et en particulier son simulateur de route https://github.com/vinzeebreak/road_simulator mais n'utilise pas la même connectique RaspberryPi-Moteurs (réduction du coût) ce qui nécessite un changement du code.

Toutes nos interfaces ont été codées en Python, ce qui est suffisant pour un traitement relativement lent des processus.


# HARDWARE

# Raspberry Pi 3/2/Zero
En fonction des coûts et des performances, on choisit de réaliser la voiture avec l'un de ces 3 Pi, de la plus performante à la moins performante.

# Moteurs/Plateforme/Roues

# Contrôle des moteurs: L293DNE et alimentation

# Camera

# Coût total




# SOFTWARE

# Initialisation de la Raspberry Pi, installation des modules

# Connexion à la Pi: Wifi/SSH

# Essais de conduite de la voiture depuis une interface python
Pour tester les branchements des moteurs aux Batterie/L293/RaspberryPi, nous avons décidé de créer un code pour conduire la voiture. Cela permet aussi de calibrer la direction et la vitesse qui peuvent comporter une partie imprévisible, surtout avec du matériel bas de gamme. Nous avons utilisé un fichier python, et le module GPIO qui contrôle le courant en sortie des pins du GPIO sur la RaspberryPi (la double rangée de pins sur la Pi). On implémente la classe Motor pour définir les 2 moteurs dans le code et les pins qui leurs correspondent en sortie du GPIO. Puis on lance la fonction race(), de préférence depuis la console ssh sur l'ordinateur pour un temps de traitement plus faible. La vitesse se contrôle avec les touches 0 à 5 (0 arrêt, 5 vitesse maximale), la direction se fait grâce aux touches q/d pour gauche/droite, z/s pour devant/derrière, en appuyant à chaque fois sur entrée pour rentrer la commande, le programme attend une entrée avec la fonction input().
Le code est le suivant:


# Création du modèle, choix des paramètres
Le modèle contient plusieurs parties: Le simulateur de route, créé par Vincent Houlbrèque (cf introduction) crée des images virtuelles de route en simulant le sol et les lignes de la route avec différents effets permettant de se rapprocher d'images réelles de la course; le réseau de neurones qui prend ce dataset en entrée et crée un modèle pouvant prédire la direction associée à une image de route (direction que la voiture devrait prendre si elle se trouvait devant l'image de route prise en compte). 
A propos du simulateur de route: Plusieurs couches (layers) peuvent être ajoutées à l'instance roadsimulator de la classe Simulator(). La couche DrawLines dessine des lignes (on peut changer une partie des arguments (couleur, etc...), et en modifiant la classe DrawLines, on peut choisir si on veut dessiner seulement la ligne centrale, ou les lignes extérieures, si on veut des lignes en plus servant de "bruit", etc...).
La couche Perspective permet d'aplatir l'image initiale pour donner un effet de perspective pour ressembler à l'image réelle prise depuis une caméra censée être proche du sol. 
La couche Crop s'applique après la couche Perspective pour reprendre l'image aplatie et la recadrer en remplissant un rectangle entier, sans bandes noires sur le côté dûes à "Perspective". Attention aux dimensions des images transformées, qui doivent être les bonnes pour le traitement par le réseau de neurone ensuite. 
En exécutant le code, on peut créer le dataset avec l'instance roadsimulator à qui on a ajouté les layers.
En ce qui concerne le modèle: C'est un réseau de neurones créé avec keras fonctionnant sur TensorFlow, avec une partie dense et une partie convolutionnelle. Il est largement inspiré de celui de Vincent Houlbrèque préconisé sur le Readme de son Git RoadSimulator. Le modèle est assez simple, il fonctionne déjà très bien sur des images de route simple (avec une seule ligne noire devant la voiture), avec une accuracy de 95%. Pour les images de route plus compliquées (plusieurs couleurs, plusieurs lignes, ajout de lignes de bruitage...) le réseau ne donne pas une très bonne accuracy, il faut le modifier ou essayer un grand nombre d'epochs (>20), ou un grand nombre d'images (>2000), ce qui prend du temps en exécution.
Les codes pour le simulateur de route sont dans le dossier road_simulator-master. Le dossier ground_pics permet de donner au modèle des images du sol sur lequel a lieu la course pour créer des images plus fidèles à la réalité. 
Le code TEST Simulator.py permet de créer un modèle du type .h5, il faudra importer cette bibliothèque sur la Pi. Le modèle est déplacé sur la Pi et chargé dans le code utilisé pour la course (avec la bibliothèque Keras) pour prédire directement la direction en utilisant l'image prise par la caméra, avec model.predict().

# Lancement de la course
Le code pour le lancement de la course est le code automatic_run.py. 
Avec la bibliothèque keras on charge le modèle pour les prédictions. La bibliothèque PiCamera installée sur la Pi permet de récupérer les photos. On a décidé d'utiliser Python pour récupérer environ deux images par seconde (en photo et pas en stream continu), ce qui est suffisant pour une voiture lente mais très peu pour une voiture rapide. L'utilisation du module OpenCV pour récupérer les photos est conseillée (nous n'avons pas réussi à télécharger le module), pour éviter Python qui est très lent pour ce genre d'opérations système. On choisit d'appeler la fonction principale avec un nombre maximal d'images prises pour que la voiture s'arrête à un moment, et on spécifie ses vitesses en ligne droite et en virage (toujours entre 0 et 5). Pour chaque image prise par la caméra, (stockée dans Current_Pic (ne pas oublier de créer ce répertoire)), on stocke en plus cette image avec les autres dans le dossier Pictures, on prédit la direction grâce au modèle chargé, et on transforme cette direction (le résultat de la prédiction du réseau est une direction discrétisée en 5 valeurs: extrême gauche, gauche, tout droit, droite, extrême droite) en une commande de vitesse pour les roues. Puis on efface l'image de Current_Pic et on recommence, tant qu'on a pris moins d'images que le maximum fixé.
Le code de lancement de la course autonome:

# CONCLUSION
Nous avons tenté de résumer notre travail pour permettre de créer facilement une voiture autonome et surtout avoir l'occasion d'expérimenter Hardware et Software ce qui procure une bonne initiation à l'électronique, Linux, les protocoles de connexion, les réseaux de neurones, etc...
Les éléments d'amélioration sont: Eviter l'utilisation de Python pour récupérer les photos de la caméra (s'inspirer du code de Vincent H); Modifier le réseau pour obtenir une bonne accuracy sur des routes plus complexes.

Pour plus d'informations, toute question, contactez baptiste MOREAU-PERNET sur ce Github.

# A ACTUALISER
(manque les files test.py pour conduire la voiture, à commenter, le file autonomous_race (à commenter))
