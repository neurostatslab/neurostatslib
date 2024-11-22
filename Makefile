# You can set these variables from the command line, and also
# from the environment for the first two.
SOURCEDIR     = notebooks
BUILDDIR      = docs/data_sets

# Put it first so that "make" without argument is like "make help".
help:
	@echo "Usage: make <notebook_name>"
	@echo ""
	@echo "Converts <notebook_name>.ipynb from 'notebooks' folder to <notebook_name>.md in 'docs/data_sets' folder"
	@echo ""
	@echo "Available notebooks:"
	@ls $(SOURCEDIR) | sed 's/\.ipynb//g' | sed 's/^/    /'


.PHONY: help Makefile

%: Makefile
	@jupytext "$(SOURCEDIR)/$@.ipynb" --to myst --output "$(BUILDDIR)/$@.md"
