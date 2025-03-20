@ECHO OFF

pushd %~d

set SOURCEDIR=docs/data_sets
set BUILDDIR=notebooks

if "%1" == "" goto help

jupytext %SOURCEDIR%\%1.md --to notebook --output %BUILDDIR%\%1.ipynb
goto end

:help
(
	echo.Usage: %0 <notebook_name>
	echo.
	echo.Convert <notebook_name>.ipynb to'notebooks' folder from <notebook_name>.md in 'docs/data_sets' folder.
)

:all(
    for %%f in (%SOURCEDIR%\*.md) do (
        jupytext %%f --to notebook --output %BUILDDIR%\%%~nf.ipynb
    )
)

:end
popd