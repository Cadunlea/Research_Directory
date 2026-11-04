# Word version

`../literature_review.docx` is generated from the LaTeX sources:

```bash
python3 flatten.py            # one pandoc-friendly flat.tex
pandoc flat.tex -f latex -t docx --citeproc --bibliography ../refs.bib \
       -M reference-section-title=References --number-sections -o raw.docx
python3 style.py              # Arial 11, gray table headers, US Letter
```

Run from a folder containing `figure_architecture.png`. Citations come out
author-year (pandoc's default style); the PDF uses numbered citations.
