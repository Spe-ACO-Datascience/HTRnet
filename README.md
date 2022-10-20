# HTRnet

Aide pour comprendre la structure du projet : 

* Le fichier principal à consulter est le fichier Main_flow.py, c'est à partir de ce fichier que sont réalisés 
tous les tests et notre démarche. Notre démarche se veut exploratoire, c'est nous avons laissé les grandes étapes
de notre projet et les différentes hypothèses et réflexions par lesquelles nous sommes passés.

* Les fichiers 
    - KNN.py
    - Linear_SVC.py
    - RF.py
    - neural_network.py
sont des scripts avec une fonction permettant d'exécuter chacun des modèles avec une estimation des différents hyper-paramètres
en cross-validation systèmatiquement.

* Le fichier utils.py héberge un ensemble de fonctions permettant de réaliser des opérations courantes pour notre projet
tel que l'affichage des prédictions. 

* Pour traiter les données brutes, nous avons réalisé un script dans le fichier "grid.py" qui nous a permis de diviser
chacun des feuilles de départ en images contenant chacune une lettre 

* le script tkteach.py n'est pas de nous, il s'agit d'un script permettant de réaliser une labelisation d'images. 

* Nous avons réalisé une base de données stockée dans le fichier storage.db, afin de lire ce type de fichier,
il faut installer une application telle que db-browser. Les requêtes se font en SQL. 

* Le fichier db-config.py permet de communiquer directement avec la base de données afin d'extraire les informations
souhaitées. 

* Tous les autres fichiers ne sont pas forcément utiles, nous les gardons afin de laisser une trace de nos réflexions durant le projet
et afin de conserver certaines idées que nous avons eu et que nous pourrions développer ultérieurement pour continuer
ce projet à titre personnel.

