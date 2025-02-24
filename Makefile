# You can set these variables from the command line, and also
# from the environment for the first two.
SOURCEDIR     = docs/data_sets
BUILDDIR      = notebooks

# Put it first so that "make" without argument is like "make help".
help:
	@echo "Usage: make <notebook_name>"
	@echo ""
	@echo "Creates <notebook_name>.ipynb in 'notebooks' folder from <notebook_name>.md in 'docs/data_sets' folder"
	@echo ""
	@echo "Available notebooks:"
	@ls $(SOURCEDIR) | sed 's/\.md//g' | sed 's/^/    /'

.PHONY: help Makefile

all:
	@echo "Creating all notebooks..."
	@for file in $(SOURCEDIR)/*.md; do \
		notebook_name=$$(basename $$file .md); \
		echo "Creating $$notebook_name.ipynb..."; \
		jupytext "$$file" --to notebook --output "$(BUILDDIR)/$$notebook_name.ipynb"; \
	done

%: Makefile
	@echo "Creating $@.ipynb...";
	@jupytext "$(SOURCEDIR)/$@.md" --to notebook --output "$(BUILDDIR)/$@.ipynb"
