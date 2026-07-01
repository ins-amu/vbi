# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = VBI
BUILDDIR      = _build



# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Live-reloading local preview: rebuilds and refreshes the browser tab
# automatically on every source change, avoiding stale cached plot images
# (sphinx-gallery output images keep a fixed filename across rebuilds, so a
# plain browser refresh after `make html` can silently serve the old one).
livehtml:
	sphinx-autobuild "." "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

.PHONY: help livehtml Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
