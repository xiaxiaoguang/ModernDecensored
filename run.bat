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
@REM FOR /L %%i IN (1,1,3) DO (
@REM     echo ----------------------------------------------------------------
@REM     echo Processing Z%%i...
@REM     echo ----------------------------------------------------------------
@REM     python work.py --input "G:\manga\Z%%i" --output "G:\manga\Z%%i-1" --mosaic 1
@REM )
@REM python work.py --input "G:\manga\Z1" --output "G:\manga\Z1-1" --mosaic 1
python work.py --input "G:\manga\Z2" --output "G:\manga\Z2-1" --mosaic 2 --mode refine
@REM python work.py --input "G:\manga\Z2" --output "G:\manga\Z2-1" --mosaic 2

@REM python work.py --input "E:\MangaTranslator\HAI\HAI\input" --output "E:\MangaTranslator\HAI\HAI\output" --mosaic 2
@REM python work.py --input "G:\manga\Z2" --output "G:\manga\Z2-1" --mosaic 1
@REM python utils/webptopdf.py "G:\manga\Z6-1" "G:\manga\Z5-1" "G:\manga\Z4-1" "G:\manga\Z3-1" "G:\manga\Z2-1" "G:\manga\Z1-3"
@REM python utils/webptopdf.py "G:\manga\Z8" "G:\manga\Z9" "G:\manga\Z10" "G:\manga\Z11" "G:\manga\Z12"
@REM python utils/webptopdf.py "G:\manga\K1" "G:\manga\K8"  "G:\manga\K4" "G:\manga\K2" "G:\manga\K3" "G:\manga\K10" "G:\manga\K9" 
@REM python utils/webptopdf.py "G:\manga\K1"