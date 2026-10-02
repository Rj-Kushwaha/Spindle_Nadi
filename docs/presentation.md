# Presentation

Create or refresh the nine-slide demo deck from the current generated dataset
and the same held-out evaluation used by the dashboard:

```sh
python -m pip install -r requirements-presentation.txt
python -m docs.create_presentation
```

The generated deck is `assets/Spindle_Nadi_Demo.pptx`. Its metric values are
recomputed from the labeled synthetic dataset; each slide explicitly limits
claims to this prototype. Use the spoken demo and judge Q&A in
[`demo.md`](demo.md) alongside the slides.
