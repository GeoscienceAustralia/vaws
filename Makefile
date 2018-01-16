# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = VAWS
SOURCEDIR     = source
# BUILDDIR      = build
BUILDDIR      = ../vaws-docs
PDFBUILDDIR   = /tmp
PDF           = ./manual.pdf

# Internal variables
PAPEROPT_letter = -D latex_paper_size=letter
PAPEROPT_a4     = -D latex_paper_size=a4
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

.PHONY: help html latexpdf

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

latexpdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(PDFBUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	make -C $(PDFBUILDDIR)/latex all-pdf
	cp $(PDFBUILDDIR)/latex/*.pdf $(PDF)
	@echo "pdflatex finished; see $(PDF)"

buildandcommithtml: html latexpdf
	cd $(BUILDDIR)/html; git add . ; git commit -m "rebuilt docs"; git push origin gh-pages

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
#%: Makefile
#	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
