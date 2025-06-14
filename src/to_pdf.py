import time
from pathlib import Path


def dump_figure_to_pdfs(fig, run_name, subgroup, measure, channel_name, plot_name='histogram', limit_size=None):
    SIZES = [
        (w, h)
        for w, h in
        [(300, 250), (400, 450), (600, 500), (800, 500), (1200, 600), (1300, 910), (800, 1200), (600, 1000)]
        if limit_size is None or (w <= limit_size and h <= limit_size)
    ]
    out_folder = Path(f"output/pdfplots/{run_name}") / plot_name / subgroup
    out_folder.mkdir(parents=True, exist_ok=True)
    fig.write_image(out_folder / "throwaway.pdf", format="pdf", width=100, height=100)  # prevents an ugly mathjax overlay being included
    time.sleep(1)
    for w, h in SIZES:
        size_folder = out_folder / f"{w}x{h}"
        size_folder.mkdir(parents=False, exist_ok=True)
        out_file_pdf = size_folder / f"{channel_name}_{measure}.pdf"
        fig.write_image(out_file_pdf, format='pdf', width=w, height=h)
        print(f"Saved plot to {out_file_pdf}")
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