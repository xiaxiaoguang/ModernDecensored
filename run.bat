@echo off
cd /d "%~dp0"
IF NOT EXIST "G:\" (
    echo 错误: 找不到 G 盘。请检查移动硬盘是否连接。
    pause
    exit
)
IF NOT EXIST "E:\" (
    echo 错误: 找不到 E 盘。请检查路径配置。
    pause
    exit
)

FOR /L %%i IN (1,1,3) DO (
    echo ----------------------------------------------------------------
    echo Processing Z%%i...
    echo ----------------------------------------------------------------
    python work.py --input "G:\manga\Z%%i" --output "G:\manga\Z%%i-1" --mosaic 1
)

python work.py --input "G:\manga\Z2" --output "G:\manga\Z2-1" --mosaic 2 --mode refine

@REM python work.py --input "E:\MangaTranslator\HAI\HAI\input" --output "E:\MangaTranslator\HAI\HAI\output" --mosaic 2
@REM python work.py --input "G:\manga\Z2" --output "G:\manga\Z2-1" --mosaic 1
@REM python utils/webptopdf.py "G:\manga\K1"
