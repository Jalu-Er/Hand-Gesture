# handgesture

`handgesture` adalah program kontrol slide presentasi berbasis gestur tangan. Program menggunakan webcam, OpenCV, MediaPipe HandLandmarker, dan PyAutoGUI untuk mengirim perintah keyboard ke Google Slides.

## Fitur

- Swipe kanan: next slide.
- Swipe kiri: previous slide.
- Jump slide dengan dua tangan.
- Hold target sekitar 1 detik sampai status `READY`.
- Konfirmasi jump dengan kepalan dua tangan.
- Freeze/unfreeze gesture dengan satu tangan di sisi kanan layar menunjukkan angka `1` selama 2 detik.
- Active gesture zone 15%-85% lebar kamera untuk mengurangi salah deteksi dari tepi frame.
- UI kamera untuk debugging: target slide, status, FPS, landmark tangan, dan status jari.

## File Penting

- `handgesture.py`: program utama.
- `hand_landmarker.task`: model pre-trained MediaPipe untuk deteksi 21 landmark tangan.
- `requirements.txt`: dependency Python.
- `handgesture.bat`: setup otomatis sekaligus menjalankan program di Windows.

## Instalasi Windows

Pastikan Python sudah terinstall. Disarankan Python 3.10 atau 3.11.

1. Clone repository:

```bash
git clone https://github.com/Jalu-Er/Hand-Gesture.git
cd Hand-Gesture
```

2. Jalankan program:

```bash
handgesture.bat
```

File `handgesture.bat` akan otomatis membuat virtual environment dan menginstall dependency jika belum tersedia, lalu menjalankan program.

## Instalasi Manual

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python handgesture.py
```

Untuk uji deteksi tanpa mengirim tombol keyboard ke slide:

```bash
python handgesture.py --dry-run
```

Jika webcam utama bukan index `0`:

```bash
python handgesture.py --camera 1
```

## Cara Menggunakan di Google Slides

1. Buka presentasi di Google Slides.
2. Masuk ke mode presentasi.
3. Pastikan window presentasi sedang aktif/fokus.
4. Jalankan `handgesture`.
5. Gunakan gestur:
   - swipe kanan untuk next slide,
   - swipe kiri untuk previous slide,
   - dua tangan untuk memilih target jump,
   - tahan target sekitar 1 detik sampai `READY`,
   - kepalkan dua tangan sekali untuk konfirmasi jump,
   - satu tangan sisi kanan layar angka `1` selama 2 detik untuk freeze/unfreeze.

## Mapping Angka Per Tangan

- `0`: kepal.
- `1`: telunjuk.
- `2`: telunjuk + tengah.
- `3`: telunjuk + tengah + manis.
- `4`: telunjuk + tengah + manis + kelingking.
- `5`: jempol.
- `6`: jempol + telunjuk.
- `7`: jempol + telunjuk + tengah.
- `8`: jempol + kelingking.
- `9`: semua jari terbuka.

Sisi kiri frame digunakan sebagai digit puluhan. Sisi kanan frame digunakan sebagai digit satuan.

## Catatan Teknis

- File `hand_landmarker.task` wajib ada di folder yang sama dengan `handgesture.py`.
- UI kamera digunakan untuk debugging/pengujian. Saat presentasi sebenarnya, layar utama tetap Google Slides.
- Program memakai normalisasi brightness ringan untuk membantu kondisi terang/redup tanpa terlalu menurunkan FPS.
- Jika swipe sulit terbaca, coba:

```bash
python handgesture.py --swipe-threshold 0.10
```

- Jika perintah terlalu sensitif, naikkan cooldown:

```bash
python handgesture.py --cooldown 1.5
```
