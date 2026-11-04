"""Flatten the LaTeX sources into one pandoc-friendly file."""
import re, pathlib
SRC = pathlib.Path("/home/user/Research_Directory/docs/literature_review")

def read(name):
    return (SRC / f"{name}.tex").read_text()

def expand(text):
    return re.sub(r"\\input\{(\w+)\}",
                  lambda m: expand(read(m.group(1))) if m.group(1) != "fig_architecture"
                  else r"\includegraphics[width=4.6in]{figure_architecture.png}", text)

body = expand(read("sec_review") + "\n" + read("sec_model"))
body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("%"))

numbers = {"tab:literature": ("Table", 1), "tab:layers": ("Table", 2),
           "tab:model": ("Table", 3), "fig:architecture": ("Figure", 1)}

def xltab(m):
    block = m.group(0)
    block = re.sub(r"\\endfirsthead.*?\\endlastfoot", "", block, flags=re.S)
    header = re.search(r"\\midrule", block)
    ncols = None
    for line in block.splitlines():
        if "\\textbf{" in line and "&" in line:
            ncols = line.count("&") + 1
            break
    block = re.sub(r"\\begin\{xltabular\}\{\\textwidth\}\{.*?\}\s*\n(?=\\caption)",
                   "\\\\begin{longtable}{" + "l" * ncols + "}\n", block, flags=re.S)
    block = block.replace("\\end{xltabular}", "\\end{longtable}")
    return block

body = re.sub(r"\\begin\{xltabular\}.*?\\end\{xltabular\}", xltab, body, flags=re.S)
# spec lines of xltabular span several lines; make sure none survived
assert "xltabular" not in body, "xltabular left over"
body = re.sub(r"\\multicolumn\{(\d+)\}\{(?:[^{}]|\{[^{}]*\})*\}\{\\textit\{([^}]*)\}\}",
              lambda m: r"\textit{" + m.group(2) + "}" + " &" * (int(m.group(1)) - 1), body)
body = re.sub(r"\\rowcolor\{[^}]*\}", "", body)
body = body.replace("@{}", "")
def squeeze(m):
    return "\n".join(l for l in m.group(0).splitlines() if l.strip())
body = re.sub(r"\\begin\{(tabular|longtable)\}.*?\\end\{\1\}", squeeze, body, flags=re.S)
body = body.replace("\\begin{table}[H]", "\\begin{table}").replace("\\begin{figure}[H]", "\\begin{figure}")

# number the captions and resolve cross-references
def caption(m):
    text, label = m.group(1), m.group(2)
    kind, n = numbers[label]
    return f"\\caption{{{kind} {n}. {text}}}\\label{{{label}}}"
body = re.sub(r"\\caption\{(.*?)\}\s*\\label\{([\w:]+)\}", caption, body, flags=re.S)
body = re.sub(r"(Table|Figure)~\\ref\{([\w:]+)\}",
              lambda m: f"{numbers[m.group(2)][0]}~{numbers[m.group(2)][1]}", body)
assert "\\ref{" not in body, re.findall(r".{30}\\ref\{.*?\}", body)

doc = r"""\documentclass{article}
\usepackage{natbib}
\title{Food-intake detection from AIM-2 sensor signals: literature review and model description}
\author{Caelan Dunlea, University of Alabama}
\date{1 October 2026}
\begin{document}
\maketitle
""" + body + "\n\\end{document}\n"
pathlib.Path("flat.tex").write_text(doc)
print("ok", len(doc))
