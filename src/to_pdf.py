import json
import time
from pathlib import Path

import torch
from src.config import get_current_config
import logging
logger = logging.getLogger(__name__)

def dump_figure_to_pdfs(fig, plot_name, subgroup, measure, channel_name, limit_size=None, metadata: dict =None):
    """Export a Plotly figure to PDF at multiple sizes and an "atom" compact variant.

    Writes to output/pdfplots/{run_name}/{plot_name}/{subgroup}/{WxH}/{channel_name}_{measure}.pdf
    for each size in [(300,250), (400,450), (600,500), (800,500), (1200,600), (1300,910),
    (800,1200), (600,1000)]. Also writes an atom variant (500x400, serif font, no margins)
    for minimal-margin thesis figures. A throwaway.pdf is written first to flush
    kaleido's MathJax overlay artifact.

    If metadata is provided, saves it as batch_metrics_{subgroup}_{channel_name}_{measure}.json
    alongside the PDFs.

    Args:
        fig: Plotly figure to export.
        plot_name: Subdirectory under the run's pdfplots folder (e.g. "qualitative_samples").
        subgroup: Further subdirectory (e.g. "full" or "nolegend").
        measure: Label appended to the filename after the channel name.
        channel_name: Label prepended to the filename (often the shot number or channel).
        limit_size: If set, only exports sizes where both width and height are <= this value.
        metadata: Optional dict of metrics to save alongside the PDFs as JSON.
            Tensors are converted to scalars; non-serializable values become strings.
    """
    run_name = get_current_config().run_name
    SIZES = [
        (w, h)
        for w, h in
        [(300, 250), (400, 450), (600, 500), (800, 500), (1200, 600), (1300, 910), (800, 1200), (600, 1000)]
        if limit_size is None or (w <= limit_size and h <= limit_size)
    ]
    out_folder = Path(f"output/pdfplots/{run_name}") / plot_name / subgroup
    out_folder.mkdir(parents=True, exist_ok=True)
    # kaleido includes a MathJax loading overlay in the first PDF it renders; this throwaway flushes it
    fig.write_image(out_folder / "throwaway.pdf", format="pdf", width=200, height=300)  # prevents an ugly mathjax overlay being included
    time.sleep(1)
    for w, h in SIZES:
        size_folder = out_folder / f"{w}x{h}"
        size_folder.mkdir(parents=False, exist_ok=True)
        out_file_pdf = size_folder / f"{channel_name}_{measure}.pdf"
        fig.write_image(out_file_pdf, format='pdf', width=w, height=h)
        print(f"Saved plot to {out_file_pdf}")
    if metadata is not None:
        save_json_friend(out_folder / f"batch_metrics_{subgroup}_{channel_name}_{measure}.json", metadata)
    out_file_pdf = out_folder / f"atom_{subgroup}_{channel_name}_{measure}.pdf"
    fig.update_layout(
        showlegend=False,
        title_text='',
        margin=dict(l=0, r=0, t=15, b=0),
        font=dict(family="serif", size=10),
    )
    # fig.update_xaxes(title_text='Heights')
    fig.write_image(out_file_pdf, format='pdf', width=500, height=400)
    print(f"Saved plot to {out_file_pdf}")

def save_json_friend(file_path, dict):
    json_file = file_path
    dict = dict.copy()
    for k,v in dict.items():
        try:
            if isinstance(v, torch.Tensor):
                dict[k] = v.item()
            else:
                json.dumps(v)
        except Exception:
            dict[k] = str(v)
    try:
        with open(json_file, "w") as f:
            json.dump(dict, f, indent=2)
        logger.info("Saved PDF plot friend JSON-friendly cache metadata to %s", json_file)
    except Exception as e:
        logger.error("Failed to save PDF plot friend JSON-friendly cache metadata: %s", e)
