# Model Directory

Place your trained YOLOv8 weights here:

```text
model/best.pt
```

If `best.pt` is missing, the Flask app runs a deterministic simulation fallback so the UI,
preprocessing, reporting, and deployment workflow remain testable.
