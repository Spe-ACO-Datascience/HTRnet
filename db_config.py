import sqlite3 as sq


# CONNECTION À LA BASE DE DONNÉE SQLITE
db = sq.connect("storage-test.db")
cursor = db.cursor()


# Fonction de requêtes à la base de données

def selectAllCategories():
    return cursor.execute("SELECT * FROM categories")


def selectImages(nbre):
    return cursor.execute(
        """
                SELECT i.imagePath, c.categoryName
                FROM images as i
                INNER JOIN labels as l ON l.image_id = i.id
                INNER JOIN categories as c ON c.id = l.category_id
                LIMIT :nbre
            """,
        {"nbre": nbre}
    )


def selectImagesOfInd(name):
    """Selection de toutes les images d'une personne"""
    return cursor.execute(
        "SELECT i.imagePath, c.categoryName FROM images as i INNER JOIN labels as l ON l.image_id = i.id INNER JOIN categories as c ON c.id = l.category_id WHERE i.imagePath LIKE ?",
        ['%'+name+'%']
    )


norroyImages = selectImagesOfInd("norroy")
for i in norroyImages:
    print(i[1])
