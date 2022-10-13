def extractImageCategory(fullCat):
    extract = fullCat.split("(")
    if(len(extract) > 1):
        return extract[1].split(")")[0]
    return extract[0]


def showRequest(request):
    for el in request:
        print(el)

# print("point".split("("))
