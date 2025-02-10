# 🛸 Drone Coordinate Tracking Project
Bu proje, bilgisayarlı görü teknikleri kullanarak bir drone'un hareketlerini takip eder. X ve Y koordinatlarındaki değişimleri hesaplar ve bu değişimleri gerçek dünya mesafelerine çevirir.

## 🚀 Özellikler
- **Optik akış (Optical Flow) kullanarak çerçeveler arasındaki merkezi kaymaları hesaplar.**
- **Kamera yüksekliği ve görüş alanı kullanılarak piksel kaymaları gerçek dünya mesafelerine dönüştürülür.**
- **Her çerçeve için X ve Y eksenindeki toplam konum değişikliği metre cinsinden hesaplanır.**

## 🛠️ Kullanılan Teknolojiler
- **Python**
- **YOLO (You Only Look Once)**
- **OpenCV**
- **NumPy**
- **Pandas**

## 📌 Gereksinimler
Aşağıdaki kütüphanelerin yüklü olduğundan emin olun:
```bash
pip install opencv-python numpy pandas
```

## ▶️ Nasıl Çalıştırılır?
Aşağıdaki komutu çalıştırarak bir video dosyası ile programı başlatabilirsiniz:
```bash
python drone_tracking.py
```

## 📜 Kod Açıklamaları
- **calculate_center_shift(p0, p1, st):** İki çerçeve arasındaki optik akış ile X ve Y yönündeki ortalama kaymayı hesaplar.
- **calculate_pixel_to_meter_ratio(altitude, fov, image_width=1920, image_height=1080):** Piksel ile metre arasındaki dönüşüm oranını hesaplar.
- **Ana Script:** Bir video dosyasını okur, optik akışı hesaplar ve drone hareketlerini takip ederek her çerçevede kaymaları metre cinsinden yazdırır.

## 📊 Örnek Çıktı
```
Frame 1 - Shift X: 0.05 meters, Shift Y: 0.03 meters. Total Position - X: 0.05 meters, Y: 0.03 meters
```

## 📜 Lisans
Bu proje **MIT Lisansı** altında paylaşılmıştır.

---
📩 **Bana ulaşın:** [t.necatgok@gmail.com](mailto:t.necatgok@gmail.com)

---
