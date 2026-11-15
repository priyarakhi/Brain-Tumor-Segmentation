# Dataset placement (not supplied in this repository)

This repository **does not** include the original brain MRI image data or `annotations_all.json`.
The notebook referred to an archive named `datas.zip`, unpacked as follows:

```text
data/raw/
└── datas/
    ├── all/
    │   ├── y0.jpg
    │   ├── y1.jpg
    │   └── ...
    └── annotations_all.json
```

The annotation export used a VIA-like JSON object whose entries included `filename` and `regions`, each
region with polygon coordinates under `shape_attributes.all_points_x` and `.all_points_y`.

Place **only data you have permission to use** here; raw data and derived masks are gitignored.
If you downloaded the original `datas.zip`, extract it manually and confirm this layout.
See `docs/DATASET.md` for integrity checks and patient-level split limitations.
