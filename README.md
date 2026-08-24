# تصنيف حالة السماء (غيوم / سماء صافية) ☁️☀️

مشروع تصنيف صور باستخدام **التعلم الآلي (Machine Learning)** يقوم بتحديد ما إذا كانت الصورة المُدخلة تُظهر **سماء ملبدة بالغيوم** أو **سماء صافية**، وذلك بالاعتماد على نموذج تم تدريبه عبر منصة **Google Teachable Machine** ثم تشغيله باستخدام مكتبة **Keras/TensorFlow** في بايثون.

---

## 📌 فكرة المشروع

تم تدريب نموذج تصنيف صور (Image Classification Model) على فئتين:

| الفئة | الوصف | عدد عينات التدريب |
|---|---|---|
| ☁️ غيوم | صور لسماء ملبدة بالغيوم | 7 صور |
| 🌤️ سماء صافية | صور لسماء صافية زرقاء | 6 صور |

تم تدريب النموذج عبر [Google Teachable Machine](https://teachablemachine.withgoogle.com/) ثم تصديره بصيغة **Keras (.h5)**، بعدها تم استخدامه للتنبؤ بصور جديدة عبر كود بايثون يعمل على **Google Colab**.

---

## 🛠️ خطوات العمل

1. **جمع البيانات**: تم رفع صور لحالتين مختلفتين للسماء (غيوم / صافية) داخل منصة Teachable Machine.
2. **تدريب النموذج**: تم تدريب النموذج مباشرة من المتصفح دون الحاجة لكتابة كود تدريب يدوي.
3. **اختبار النموذج (Preview)**: تم التحقق من دقة النموذج مباشرة داخل المنصة، حيث نجح في تصنيف صورة غيوم بنسبة ثقة **100%**.
4. **تصدير النموذج**: تم تصدير النموذج بصيغة Keras للحصول على ملفين أساسيين:
   - `keras_model.h5` → النموذج المدرَّب.
   - `labels.txt` → أسماء الفئات (الأصناف).
5. **تشغيل النموذج على صورة جديدة**: تم كتابة كود بايثون (يعمل على Google Colab) لتحميل النموذج والتنبؤ بصورة جديدة (`image.png`).

---

## 📂 محتويات المجلد

```
📁 تسليم التاسك
 ├── README.md                    # هذا الملف (شرح المشروع والكود)
 ├── keras_model.h5               # النموذج المدرَّب المُصدَّر من Teachable Machine
 ├── labels.txt                   # أسماء الفئات (الأصناف)
 ├── Teachable Machine.png        # صورة من واجهة Teachable Machine أثناء التدريب والمعاينة
 └── colab.png                    # صورة من Google Colab توضح تشغيل الكود والنتيجة
```

> ⚠️ ملاحظة: ملفا `keras_model.h5` و `labels.txt` مرفقان فعليًا في هذا المجلد. الشيء الوحيد الناقص لتشغيل الكود هو **صورة الاختبار** (مثال: `image.png`) — ضِفها بنفس اسمها في الكود، أو عدّل اسم الملف داخل السطر `Image.open("image.png")`.

---

## 💻 الكود

```python
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

import tf_keras as tk

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tk.models.load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("image.png").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
```

---

## ▶️ طريقة التشغيل

يمكن تشغيل المشروع بسهولة عبر **Google Colab**:

1. افتح [Google Colab](https://colab.research.google.com/) وأنشئ ملف Notebook جديد.
2. ارفع الملفات التالية إلى بيئة العمل (Files):
   - `keras_model.h5`
   - `labels.txt`
   - صورة الاختبار (وسمّها `image.png` أو عدّل اسم الملف في الكود).
3. ثبّت المكتبة المطلوبة داخل الـ Notebook:
   ```bash
   pip install tf_keras
   ```
4. الصق الكود أعلاه في خلية جديدة وشغّله.
5. ستظهر النتيجة على الشكل التالي:
   ```
   Class: غيوم
   Confidence Score: 0.9995708
   ```

### التشغيل محليًا (اختياري)

```bash
pip install tensorflow tf_keras pillow numpy
python predict.py
```

---

## 📦 المتطلبات (Requirements)

- Python 3.x
- tensorflow
- tf_keras
- pillow (PIL)
- numpy

---

## 📊 النتيجة

عند اختبار النموذج بصورة تحتوي على غيوم، أعطى النموذج تصنيفًا صحيحًا:

```
Class: غيوم
Confidence Score: 0.9995708
```

مما يدل على دقة عالية جدًا في التمييز بين الفئتين.

---

## 📚 المصادر

- [Google Teachable Machine](https://teachablemachine.withgoogle.com/) — المنصة المستخدمة لتدريب النموذج وتصديره.
- [Google Colab](https://colab.research.google.com/) — البيئة المستخدمة لتشغيل الكود واختبار النموذج.

---
