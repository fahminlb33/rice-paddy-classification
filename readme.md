# Skripsi

Akses aplikasi: [Klik disini](http://skripsi.kodesiana.com/)

CI/CD status: [![Docker Publish](https://github.com/fahminlb33/skripsi/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/fahminlb33/skripsi/actions/workflows/docker-publish.yml)

> Klasifikasi Penyakit pada Tumbuhan Padi menggunakan Transfer Learning
> MobileNetV2 Terbantu Gradient-weighted Class Activation Mapping (Grad-CAM)

Repositori ini berisi kode sumber untuk skripsi saya dengan judul di atas, termasuk *Jupyter Notebook*, arsitektur *deep learning*, dan dataset untuk membangun model klasifikasi citra.

Dataset yang digunakan dapat diakses pada:

- [Rice Leaf Disease Image Samples](https://data.mendeley.com/datasets/fwcj7stb8r/1) (Sethy, 2020)
- Untuk dataset untuk kelas *healthy* merupakan hasil kurasi mandiri oleh penulis. Untuk mendapatkan akses, Anda bisa hubungi penulis melalui surel ke fahminlb33 [at] gmail [dot] com.

Untuk melakukan *modelling* Anda dapat menjalankan kode pada *notebook* `train-mobilenetv2.ipynb` untuk menghasilkan model TensorFlow dan *dictionary* pemetaan kelas hasil prediksi dalam format `joblib`.

## Continuous Deployment (CI/CD)

Repositori ini telah terintegrasi dengan Github Actions dan Azure App Service sehingga setiap kali ada perubahan pada repositori ini, sistem akan secara otomatis disinkronasi ke server.
