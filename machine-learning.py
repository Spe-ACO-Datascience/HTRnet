from db_config import *
from utils import *
import cv2

images = selectImages(10)
for i in images:
    img = cv2.imread(i[0])
    cv2.imshow(f'{extractImageCategory(i[1])}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
