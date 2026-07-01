# Final Report Overleaf Package

Upload this folder's contents to Overleaf:

- `main.tex`
- `acl.sty`
- `acl_natbib.bst`

Set `main.tex` as the main document.

`Vision_Guided_Autonomous_Drone_Racing_with_Privileged_Teacher_Imitation.pdf`
is the rendered report snapshot saved before local project archival. The LaTeX
source remains authoritative if it differs from that snapshot.

The report uses an inline `thebibliography` environment, so no `.bib` file is required. `acl_natbib.bst` is included because the ACL style references it during compilation.

The ACL style is loaded with:

```tex
\usepackage[preprint]{acl}
```
