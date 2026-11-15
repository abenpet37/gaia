@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 goto use_default_python

py -3.14 -c "import sys; print('%s.%s' % sys.version_info[:2])" 2>nul | findstr /x "3.14" >nul
if errorlevel 1 goto use_default_python

py -3.14 "%~dp0start.py" %*
exit /b %errorlevel%

:use_default_python
where python >nul 2>nul
if errorlevel 1 goto open_map

python "%~dp0start.py" %*
if errorlevel 1 goto open_map
exit /b 0

:open_map
echo Python initialization unavailable; opening the map without preparation.
start "" "%~dp0map\map.html"
exit /b %errorlevel%
