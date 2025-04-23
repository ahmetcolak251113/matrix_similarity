import cv2 # Resmi dosya olarak okuyup üç boyutlu bir matris oluşturmak için
import math
import warnings
import numpy as np
import PIL.Image as Image
warnings.filterwarnings('ignore')
# Görsellleri matrise dönüştürmek için imread fonksiyonu kullanılır
gorsel_1 = cv2.imread("görsel1.jpg")
gorsel_2 = cv2.imread("görsel2.jpg")

""" Görselleri matrise dönüştürdük. Boyutlarını öğrenmek için shape fonksiyonunu kullanıyoruz."""
print("Görsel 1 boyutu: ", gorsel_1.shape)
print("Görsel 2 boyutu: ", gorsel_2.shape)
# Küçük olanı büyük olanın boyutuna göre yeniden boyutlandır
if gorsel_1.shape != gorsel_2.shape:
    hedef_boyut = (min(gorsel_1.shape[1], gorsel_2.shape[1]), min(gorsel_1.shape[0], gorsel_2.shape[0]))
    gorsel_1 = cv2.resize(gorsel_1, hedef_boyut)
    gorsel_2 = cv2.resize(gorsel_2, hedef_boyut)

def normalize_vector(vector):
    norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    if norm == 0:
        return [0, 0, 0]
    else:
        return [vector[0] / norm, vector[1] / norm, vector[2] / norm]

def cosine_similarity(vector1, vector2):
    vector1 = normalize_vector(vector1)
    vector2 = normalize_vector(vector2)
    return vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]

def is_similar(vector1, vector2, threshold=0.99):
    return cosine_similarity(vector1, vector2) >= threshold

benzer = 0
toplam = gorsel_1.shape[0] * gorsel_1.shape[1]

for i in range(gorsel_1.shape[0]):
    for j in range(gorsel_1.shape[1]):
        pixel1 = gorsel_1[i][j]
        pixel2 = gorsel_2[i][j]
        if is_similar(pixel1, pixel2):
            benzer += 1
        else:
            continue
ratio = benzer /toplam * 100
print(f"Benzerlik oranı: %{ratio:.2f}")

def similarityMap():
    harita = np.zeros((gorsel_1.shape[0], gorsel_1.shape[1], 3), dtype=np.uint8)
    for i in range(gorsel_1.shape[0]):
        for j in range(gorsel_1.shape[1]):
            pixel1 = gorsel_1[i][j]
            pixel2 = gorsel_2[i][j]
            if is_similar(pixel1, pixel2):
                harita[i, j] = pixel1  # benzerse orijinal piksel
            else:
                harita[i, j] = [128, 128, 128]  # değilse gri ton
    return harita

harita = similarityMap()
Image.fromarray(harita).save("benzerlik_haritasi.png")

