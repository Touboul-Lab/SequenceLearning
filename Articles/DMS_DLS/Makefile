FIGURES_PDF := $(addprefix Figures/phd/, Figure_0.pdf Figure_1.pdf Figure_2.pdf Figure_S1.pdf Figure_S2.pdf)

main.pdf : main.tex phd/Abstract.tex phd/Discussion.tex phd/Intro.tex phd/Methods.tex phd/Results.tex phd/Exp.tex phd/A-Fig.tex ref.bib $(FIGURES_PDF)
	pdflatex main
	biber main
	pdflatex main
	pdflatex main

	rm -f *.aux *.bbl *.blg *.log *.run.xml *.toc *.bcf *.out

	open -a Preview.app main.pdf

Figures/phd/Figure_1.pdf : Figures/phd/Figure_1.svg Figures/phd/Setup.svg Figures/phd/STDP.svg
	/Applications/Inkscape.app/Contents/MacOS/inkscape $(CURDIR)/Figures/phd/Figure_1.svg -C --export-type=pdf &>/dev/null

Figures/phd/Setup.svg : Code/main_figure_article.py Code/FiguresDMS.py
	python Code/main_figure_article.py

Figures/phd/Figure_0.pdf : Figures/phd/Figure_0.svg
	/Applications/Inkscape.app/Contents/MacOS/inkscape $(CURDIR)/Figures/phd/Figure_0.svg -C --export-type=pdf &>/dev/null

Figures/phd/Figure_2.pdf : Figures/phd/Figure_2.svg
	/Applications/Inkscape.app/Contents/MacOS/inkscape $(CURDIR)/Figures/phd/Figure_2.svg -C --export-type=pdf &>/dev/null

Figures/phd/Figure_S1.pdf : Figures/phd/Figure_S1.svg
	/Applications/Inkscape.app/Contents/MacOS/inkscape $(CURDIR)/Figures/phd/Figure_S1.svg -C --export-type=pdf &>/dev/null

Figures/phd/Figure_S2.pdf : Figures/phd/Figure_S2.svg
	/Applications/Inkscape.app/Contents/MacOS/inkscape $(CURDIR)/Figures/phd/Figure_S2.svg -C --export-type=pdf &>/dev/null