# PlantGuard AI — Domates Yaprağı Hastalığı Sınıflandırma

PlantGuard AI, domates yaprağı fotoğraflarını üç sınıfta değerlendiren bir CNN (Evrişimli Sinir Ağı) modeli ve Streamlit tabanlı fotoğraf yükleme arayüzüdür. Model, görüntüye göre sınıf tahmini ve bir model skoru üretir. **Bu skor kalibre edilmiş güven olasılığı veya kesin tarımsal teşhis değildir.**

## Özellikler

- Domates yaprağı görüntüsünden sınıf tahmini
- Sağlıklı, erken yanıklık ve geç yanıklık sınıfları
- PyTorch ile CNN eğitimi ve test değerlendirmesi
- Eğitim grafikleri, sınıf dağılımı ve karmaşıklık matrisi
- Streamlit üzerinden görsel yükleme ve sonuç gösterimi

## Veri seti

Kaynak: [PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset), `raw/color` altındaki domates sınıfları. Kullanım öncesinde veri setinin lisans ve atıf koşullarını kontrol edin.

| Sınıf | Görüntü |
| --- | ---: |
| `Tomato___Early_blight` | 1.000 |
| `Tomato___healthy` | 1.591 |
| `Tomato___Late_blight` | 1.909 |
| **Toplam** | **4.500** |

Görüntüler depo boyutunu artırmamak için GitHub'a eklenmez. Yerel klasör düzeni:

```text
data/plantvillage/
├── Tomato___Early_blight/
├── Tomato___healthy/
└── Tomato___Late_blight/
```

## Yöntem

- Eğitim/doğrulama/test: 3.150 / 675 / 675 (%70 / %15 / %15)
- Girdi: 128 × 128 RGB
- Veri artırma: rastgele yatay çevirme ve ±15° döndürme
- Mimari: 3 Conv2D + ReLU + MaxPool2D bloğu, global average pooling, dropout, doğrusal sınıflandırıcı
- Kayıp: CrossEntropyLoss; optimizasyon: Adam, öğrenme oranı 0,001
- Eğitim: 12 epoch; batch size: 32
- Değerlendirme: accuracy, macro precision/recall/F1 ve confusion matrix

## Test sonuçları

675 görüntülük test kümesinde:

| Metrik | Sonuç |
| --- | ---: |
| Accuracy | **%91,41** |
| Macro precision | %91,32 |
| Macro recall | %89,20 |
| Macro F1 | **%90,02** |

| Sınıf | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Erken yanıklık | %88,89 | %74,67 | %81,16 |
| Sağlıklı | %97,54 | %99,58 | %98,55 |
| Geç yanıklık | %87,54 | %93,36 | %90,36 |

Model 675 test örneğinin 617'sini doğru sınıflandırmıştır. En belirgin hata, 37 erken yanıklık görüntüsünün geç yanıklık olarak tahmin edilmesidir.

## Grafikler

![Eğitim ve doğrulama grafikleri](outputs/04_egitim_grafikleri.png)

![Karmaşıklık matrisi](outputs/05_karmasiklik_matrisi.png)

Diğer çıktılar: `outputs/01_sinif_dagilimi.png`, `outputs/02_ornek_goruntuler.png`, `outputs/03_goruntu_boyutlari.png`, `outputs/metrics.json`.

## Yerel kurulum

Python 3.11'in kararlı bir sürümü önerilir. Windows PowerShell'de proje kök dizininden:

```powershell
py -3.11 -m venv .venv_new
.\.venv_new\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Modeli yeniden eğitmek için veri klasörlerini yukarıdaki yapıda yerleştirin:

```powershell
python -m src.pipeline --epochs 12
```

Uygulamayı başlatmak için:

```powershell
python -m streamlit run app.py
```

**Not:** Eğitimden oluşan model ağırlıkları `outputs/*.pt` GitHub'a eklenmez. Uygulamanın başka bir bilgisayarda tahmin yapabilmesi için önce model eğitilmeli veya ağırlıklar ayrı, erişilebilir bir kaynaktan sağlanmalıdır.

## Proje yapısı

```text
PlantGuard_AI/
├── app.py
├── src/pipeline.py
├── data/
├── outputs/
├── docs/
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore
```

## Sınırlamalar

PlantVillage görüntüleri kontrollü koşullarda çekilmiştir; farklı saha ışığı, arka plan ve kamera koşullarında performans düşebilir. Aynı bitkinin benzer görüntülerinin farklı veri bölümlerine düşmesi veri sızıntısı riski yaratabilir. Sonuçlar kesin hastalık teşhisi olarak kullanılmamalıdır.

## Yapay zekâ desteği

ChatGPT proje planlama, kod taslağı, hata ayıklama, CNN eğitimi ve dokümantasyon aşamalarında destek amaçlı kullanılmıştır. Modelin eğitim ve test sonuçları yerel çalıştırmada ölçülmüştür.

## Teslim

- GitHub deposu: Yayınlandıktan sonra bağlantı eklenecek.
- Canlı uygulama: Yayınlanırsa bağlantı eklenecek.
- Proje raporu: `docs/` klasöründe paylaşılabilir.
