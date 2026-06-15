@echo off
setlocal

cd /d "%~dp0"

if not exist hand_landmarker.task (
    echo [handgesture] File hand_landmarker.task tidak ditemukan.
    echo Pastikan file model ada di folder yang sama dengan handgesture.py.
    pause
    exit /b 1
)

if not exist .venv\Scripts\python.exe (
    echo [handgesture] Virtual environment belum ada. Membuat .venv...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Gagal membuat virtual environment.
        echo Pastikan Python sudah terinstall dan command py tersedia.
        pause
        exit /b 1
    )

    echo [handgesture] Menginstall dependency...
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Instalasi dependency gagal.
        pause
        exit /b 1
    )
) else (
    call .venv\Scripts\activate.bat
)

echo [handgesture] Menjalankan program...
python handgesture.py
pause
