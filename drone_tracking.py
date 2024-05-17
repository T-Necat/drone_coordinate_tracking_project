"""
# bu kod videodaki bütün cisimlerin hareketini algılıyor
import cv2
import numpy as np

# Video kaynağını aç
cap = cv2.VideoCapture('/Users/tng/Code/Merküt_Code/merküt_model_test/2021 Örnek Video kopyası.mp4')

# ShiTomasi köşe tespiti parametreleri
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Lucas-Kanade optical flow parametreleri
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Rastgele renkler
color = np.random.randint(0, 255, (100, 3))

# İlk kareyi oku
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Maske oluştur
mask = np.zeros_like(old_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical flow hesapla
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # İyi noktaları seç
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Hareket çiz
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
"""


"""
def calculate_pixel_to_meter_ratio(altitude, fov, image_width=1920, image_height=1080):
    
    #Bir pikselin kaç metreye denk geldiğini hesaplar.

    #Parametreler:
    #altitude (float): Kameranın yerden yüksekliği (metre cinsinden).
    #fov (float): Kameranın yatay görüş açısı (derece cinsinden).
    #image_width (int): Görüntü genişliği (piksel cinsinden). Varsayılan 1920.
    #image_height (int): Görüntü yüksekliği (piksel cinsinden). Varsayılan 1080.

    #Dönüş:
    #(float, float): Piksel başına metre oranı (x yönü, y yönü).
    
    # FOV'u radian'a çevir
    fov_rad = np.deg2rad(fov)

    # Görüntü alanının genişliğini hesapla (altitude / tan(FOV/2))
    image_width_meters = 2 * altitude * np.tan(fov_rad / 2)
    image_height_meters = image_width_meters * (image_height / image_width)

    # Piksel başına metre oranını hesapla
    pixel_to_meter_x = image_width_meters / image_width
    pixel_to_meter_y = image_height_meters / image_height

    return pixel_to_meter_x, pixel_to_meter_y


"""







"""
# Kara kare X ve Y koordinatlarını metre cinsinden değişimini verir
import cv2
import numpy as np


def calculate_center_shift(p0, p1, st):
    #Calculate the shift in the center based on good points.
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    shift_x = np.mean(good_new[:, 0] - good_old[:, 0])
    shift_y = np.mean(good_new[:, 1] - good_old[:, 1])

    return shift_x, shift_y


# Video kaynağını aç
cap = cv2.VideoCapture('/Users/tng/Code/Merküt_Code/merküt_model_test/2021 Örnek Video kopyası.mp4')

# ShiTomasi köşe tespiti parametreleri
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Lucas-Kanade optical flow parametreleri
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# İlk kareyi oku
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Başlangıç koordinatları
x0, y0 = 0.0, 0.0
x_shift_total, y_shift_total = 0.0, 0.0

# Pixel to meter dönüşüm oranı (örneğin)
pixel_to_meter_ratio = 0.05

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical flow hesapla
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # İyi noktaları seç
    shift_x, shift_y = calculate_center_shift(p0, p1, st)
    x_shift_total += shift_x
    y_shift_total += shift_y

    # Metre cinsinden değişim
    x_shift_meter = shift_x * pixel_to_meter_ratio
    y_shift_meter = shift_y * pixel_to_meter_ratio

    # Toplam pozisyon
    x_total_meter = x0 + x_shift_total * pixel_to_meter_ratio
    y_total_meter = y0 + y_shift_total * pixel_to_meter_ratio

    print(
        f"Frame {cap.get(cv2.CAP_PROP_POS_FRAMES)} - Shift X: {x_shift_meter:.2f} meters, Shift Y: {y_shift_meter:.2f} meters")
    print(f"Total Position - X: {x_total_meter:.2f} meters, Y: {y_total_meter:.2f} meters\n")

    old_gray = frame_gray.copy()
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

cap.release()
cv2.destroyAllWindows()
"""

import cv2
import numpy as np


def calculate_center_shift(p0, p1, st):
    good_new = p1[st == 1]  #İkinci karede başarıyla takip edilen noktaları seçer.
    good_old = p0[st == 1]  #İlk karede başarıyla takip edilen noktaları seçer.

    shift_x = np.mean(good_new[:, 0] - good_old[:, 0])  # X yönündeki kaymanın ortalamasını hesaplar.
    shift_y = np.mean(good_new[:, 1] - good_old[:, 1])  # Y yönündeki kaymanın ortalamasını hesaplar.

    return shift_x, shift_y # X ve Y yönündeki kaymaları döner.


def calculate_pixel_to_meter_ratio(altitude, fov, image_width=1920, image_height=1080):
    """
    Bir pikselin kaç metreye denk geldiğini hesaplar.

    Parametreler:
    altitude (float): Kameranın yerden yüksekliği (metre cinsinden).
    fov (float): Kameranın yatay görüş açısı (derece cinsinden).
    image_width (int): Görüntü genişliği (piksel cinsinden). Varsayılan 1920.
    image_height (int): Görüntü yüksekliği (piksel cinsinden). Varsayılan 1080.

    Dönüş:
    (float, float): Piksel başına metre oranı (x yönü, y yönü).
    """
    # FOV'u radian'a çevir
    fov_rad = np.deg2rad(fov)

    # Görüntü alanının genişliğini hesapla (altitude / tan(FOV/2))

    # Görüntü alanının yer seviyesindeki genişliğini hesaplar.
    image_width_meters = 2 * altitude * np.tan(fov_rad / 2) # -> FOV'un yarısının tanjantını hesaplar.

    #Aşağıdaki oranı kullanarak yer seviyesindeki yüksekliği hesaplar.
    image_height_meters = image_width_meters * (image_height / image_width) # image_height / image_width: Görüntünün boy-en oranını hesaplar.

    # Piksel başına metre oranını hesapla
    pixel_to_meter_x = image_width_meters / image_width
    pixel_to_meter_y = image_height_meters / image_height

    return pixel_to_meter_x, pixel_to_meter_y


# Video kaynağını aç
cap = cv2.VideoCapture('/Users/tng/Code/Merküt_Code/merküt_model_test/2021 Örnek Video kopyası.mp4')

# ShiTomasi köşe tespiti
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Lucas-Kanade optical flow parametreleri
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# İlk kareyi oku
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Başlangıç koordinatları
x0, y0 = 0.0, 0.0
x_shift_total, y_shift_total = 0.0, 0.0

# Pixel to meter dönüşüm oranı (örneğin) bunu girdi olarak alırsak fonksiyonu her duruma yönelik çalıştırabiliriz
altitude = 50  # Hava aracı yüksekliği (metre)
fov = 90  # Görüş açısı (derece)
pixel_to_meter_x, pixel_to_meter_y = calculate_pixel_to_meter_ratio(altitude, fov)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical flow hesapla
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # İyi noktaları seç
    shift_x, shift_y = calculate_center_shift(p0, p1, st)
    x_shift_total += shift_x
    y_shift_total += shift_y

    # Metre cinsinden değişim
    x_shift_meter = shift_x * pixel_to_meter_x
    y_shift_meter = shift_y * pixel_to_meter_y

    # Toplam pozisyon
    x_total_meter = x0 + x_shift_total * pixel_to_meter_x
    y_total_meter = y0 + y_shift_total * pixel_to_meter_y

    print(
        f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} - Shift X: {x_shift_meter:.2f} meters, Shift Y: {y_shift_meter:.2f} meters")
    print(f"Total Position - X: {x_total_meter:.2f} meters, Y: {y_total_meter:.2f} meters\n")

    old_gray = frame_gray.copy()
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

cap.release()
cv2.destroyAllWindows()
