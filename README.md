# faceCheck

Laptop'ın kamerasıyla ekranın karşısındaki kişileri izler. Bilgisayarın sahibi ekranın başında değilken **yabancı biri tek başına ekrana bakarsa ekranı otomatik kapatır.** Uygulama Windows, macOS ve Linux'ta aynı kodla çalışır.

| Durum | Ekran |
|---|---|
| Sahip kadrajda (yanında başkaları olsa da) | **Açık** |
| Sadece yabancı(lar) ekrana bakıyor | **~1,5 sn sonra kapanır** |
| Yabancı var ama ekrana bakmıyor / uzakta | Açık |
| Kimse yok | Açık (isteğe bağlı: `--absent-delay 30`) |
| Sahip geri döndü | Otomatik açılır (`overlay` / `off` modları) |

Yüz bulma için OpenCV **YuNet**, yüz tanıma için **SFace** kullanılır (ONNX). dlib veya GPU gerekmez. Modeller ilk çalıştırmada `~/.screenguard/models` klasörüne otomatik iner. Tüm işlem yerelde yapılır, görüntü hiçbir yere gönderilmez.

## Kurulum

```bash
pip install -r requirements.txt
```

Linux'ta tkinter yoksa: `sudo apt install python3-tk`

## 1) Sahibi kaydet

```bash
python screenguard.py enroll
```

Kameraya bakın. Ekranda yazan yönlere başınızı hafifçe çevirin. Program 20 örnek alır. Farklı bir ışıkta ya da gözlükle tekrar eklemek için:

```bash
python screenguard.py enroll --append
```

## 2) Korumayı başlat

```bash
python screenguard.py run --preview
```

İlk seferde `--preview` ile deneyip skorlara bakın. Yeşil kutu sahip, kırmızı kutu ekrana bakan yabancı, turuncu kutu bakmayan yabancı demektir. Ayar tamamsa önizlemesiz çalıştırın:

```bash
python screenguard.py run
```

### Modlar

| `--mode` | Ne yapar | Geri açılma |
|---|---|---|
| `overlay` (varsayılan) | Tüm ekranı kaplayan siyah perde | Sahip görününce otomatik |
| `lock` | İşletim sisteminin kilit ekranı | Şifre ile (en güvenli) |
| `off` | Monitörü kapatır (Win: SC_MONITORPOWER, macOS: pmset, Linux/X11: xset dpms) | Sahip görününce otomatik |

### Önemli ayarlar

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--threshold` | 0.40 | Sahip eşleşme eşiği. Sizi tanımıyorsa düşürün (0.36), yabancıyı sahip sanıyorsa yükseltin (0.45–0.5). |
| `--stranger-delay` | 1.5 | Yabancı kaç saniye bakınca kapansın |
| `--owner-grace` | 2.0 | Sahip kadrajdan çıkınca kaç saniye daha "var" sayılsın |
| `--absent-delay` | 0 | Kimse yoksa kapanma süresi (0 = kapalı) |
| `--min-face` | 0.08 | Uzaktaki yüzleri yok say (yüz genişliği / görüntü genişliği) |
| `--pin 1234` | – | `overlay` modunda perde açıkken PIN + Enter ile 60 sn aç |
| `--camera 1` | 0 | Başka bir kamera |

## Açılışta otomatik başlatma

- **Windows:** `Win+R` → `shell:startup` klasörüne `pythonw C:\yol\screenguard.py run` çalıştıran bir kısayol koyun.
- **macOS:** Sistem Ayarları → Genel → Giriş Öğeleri'ne ekleyin ya da bir LaunchAgent yazın. İlk çalıştırmada kamera izni verin.
- **Linux:** `~/.config/autostart/` içine bir `.desktop` dosyası koyun.

## Sınırlamalar

- Basılı fotoğraf veya ekrandaki bir video ile kandırılabilir (canlılık testi yok). Gerçek güvenlik için `--mode lock` kullanın.
- `overlay` yalnızca görsel bir perdedir; klavye kilidi değildir. Linux'ta Wayland oturumlarında `xset` çalışmaz, `overlay` ya da `lock` modunu kullanın.
- Karanlık ortamda tespit zayıflar.

## Test

```bash
pip install pytest
python -m pytest -q tests
```
