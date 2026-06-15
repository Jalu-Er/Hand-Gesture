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

    call .venv\Scripts\activate.bat

    echo.
    echo [handgesture] Update pip bersifat opsional.
    echo Biasanya dependency tetap bisa diinstall tanpa update pip.
    set /p UPDATE_PIP="Update pip sekarang? (y/N): "
    if /I "%UPDATE_PIP%"=="Y" (
        echo [handgesture] Mengupdate pip...
        python -m pip install --upgrade pip
        if errorlevel 1 (
            echo Update pip gagal. Instalasi dependency akan tetap dicoba.
        )
    )

    echo [handgesture] Menginstall dependency...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Instalasi dependency gagal.
        echo Jika error berkaitan dengan pip terlalu lama, jalankan ulang file ini dan pilih Y saat ditanya update pip.
        pause
        exit /b 1
    )
) else (
    call .venv\Scripts\activate.bat
)

echo [handgesture] Menjalankan program...
python handgesture.py
pause
