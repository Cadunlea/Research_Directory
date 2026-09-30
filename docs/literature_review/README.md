# Literature review, model description and study design

`literature_review.pdf` is the compiled document. To edit it in Overleaf, upload
every `.tex` file and `refs.bib`, then set **Menu > Compiler > XeLaTeX**.

| File | Contents |
|---|---|
| `literature_review.tex` | Main file: preamble (Arial 11 pt, grayscale) and section order |
| `sec_review.tex` | Section 1, the literature review, with Table 1 |
| `sec_model.tex` | Section 2, the model description |
| `fig_architecture.tex` | Figure 1, the architecture (TikZ). Never scale it, or the text stops being 11 pt |
| `tab_layers.tex`, `tab_model.tex` | Table 2 (layer dimensions), Table 3 (every design decision with its literature) |
| `sec_plan.tex` | Section 3, the proposed study design, with the ablation and experiment tables |
| `refs.bib` | References. A comment marks each field that still needs checking against the PDF |
| `figure_architecture.*` | Standalone figure, for slides and email |

Arial is used when installed. Otherwise Liberation Sans or TeX Gyre Heros is
used; both have the same metrics as Arial.
