from insightface.app import FaceAnalysis
import cv2
from sklearn.metrics.pairwise import cosine_similarity

app = FaceAnalysis()
app.prepare(ctx_id=0)

img1 = cv2.imread('dataset/01_HiepChu/01_HiepChu_001.jpg')
img2 = cv2.imread('dataset/03_HungChu/03_HungChu_001.jpg')

faces1 = app.get(img1)
faces2 = app.get(img2)

emb1 = faces1[0].normed_embedding
emb2 = faces2[0].normed_embedding

similarity = cosine_similarity([emb1], [emb2])[0][0]
if similarity > 0.6:
    print("giống nhau")
else:
    print("không giống nhau")