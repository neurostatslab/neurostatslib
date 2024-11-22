@ECHO OFF

pushd %~d

set SOURCEDIR=notebooks
set BUILDDIR=docs/data_sets

if "%1" == "" goto help

jupytext %SOURCEDIR%/%1.ipynb --to myst --output %BUILDDIR%/%1.md
goto end

:help
(
	echo.Usage: %0 <notebook_name>
	echo.
	echo.Convert <notebook_name>.ipynb from 'notebooks' folder to <notebook_name>.md in 'docs/data_sets' folder.
)

:end
popd