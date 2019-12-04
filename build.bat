@echo off
setlocal ENABLEDELAYEDEXPANSION

for %%V in (14,12,11,10) do if exist "!VS%%V0COMNTOOLS!" call "!VS%%V0COMNTOOLS!..\..\VC\vcvarsall.bat" amd64 && goto compile
echo Unable to detect Visual Studio path!
goto error

:compile

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
pushd src

cl main.cpp geometry.cpp timer.cpp lbfgs\lbfgs.c gui_GL.cpp glew32.lib user32.lib kernel32.lib shell32.lib gdi32.lib ole32.lib comdlg32.lib opengl32.lib glu32.lib gui.lib /wd4312 /wd4477 /I"chromium" /I"lbfgs" /Fe"..\bin\skinner.exe" /Zi /Ox /Oy /Gy /GL /openmp /fp:fast /EHsc /nologo /link /SUBSYSTEM:CONSOLE /OPT:REF || goto error

popd
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

goto :EOF

:error
popd
echo FAILED
@%COMSPEC% /C exit 1 >nul
