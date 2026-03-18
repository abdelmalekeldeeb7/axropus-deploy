#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


@dataclass
class MixedPoint:
    workload: str
    warm_hit_ratio: float
    savings_pct: float


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mixed(path: Path) -> List[MixedPoint]:
    rows: List[MixedPoint] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                MixedPoint(
                    workload=str(row["workload"]),
                    warm_hit_ratio=float(row["warm_hit_ratio"]),
                    savings_pct=float(row["savings_pct"]),
                )
            )
    return rows


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _fmt_ms(value: float) -> str:
    return f"{value:,.2f} ms"


def _build_value_prop(summary: dict) -> str:
    long_s = summary["long"]
    return (
        "Axropus removes repeated prefix compute so long-context queries that take "
        f"{_fmt_ms(float(long_s['cold_total_ms']))} on first run drop to "
        f"{_fmt_ms(float(long_s['warm_total_ms_avg']))} on replay "
        f"({_fmt_pct(float(long_s['e2e_savings_pct']))} lower end-to-end latency)."
    )


def _write_technical_explanation(path: Path, summary: dict, hardware_note: str) -> None:
    long_s = summary["long"]
    short_s = summary["short"]
    text = f"""# Technical Explanation

## What the runtime does
1. The runtime receives a prompt and deterministic generation settings.
2. It computes a stable key for the reusable prefix section.
3. On first request (cache miss), it runs normal prefill and decode, then stores the prefix state.
4. On later requests with the same prefix (cache hit), it restores that saved state and skips prefix compute.
5. Decode still runs normally, so output generation remains model-driven and deterministic.

## Why latency drops
- Prefill is the expensive part for long-context prompts.
- Skipping prefill removes most of the first-run cost on repeated context queries.

## What was measured
- Test environment: {hardware_note}
- Long-context workload: 120k prompt tokens + 128 output tokens
  - Runs: {int(long_s['runs'])}
  - Warm replay hit rate: {_fmt_pct(float(long_s['hit_rate']) * 100.0)}
  - Cold total: {_fmt_ms(float(long_s['cold_total_ms']))}
  - Warm average total: {_fmt_ms(float(long_s['warm_total_ms_avg']))}
  - End-to-end reduction: {_fmt_pct(float(long_s['e2e_savings_pct']))}
- Short-context workload: 4k prompt tokens + 128 output tokens
  - Runs: {int(short_s['runs'])}
  - End-to-end reduction: {_fmt_pct(float(short_s['e2e_savings_pct']))}

## Scope note
These measurements are from controlled replay benchmarks and should be validated with your own production traffic mix.
"""
    path.write_text(text, encoding="utf-8")


def _write_value_prop(path: Path, summary: dict) -> None:
    path.write_text(_build_value_prop(summary) + "\n", encoding="utf-8")


def _write_performance_chart_svg(path: Path, mixed_rows: List[MixedPoint]) -> None:
    long_pts = sorted([r for r in mixed_rows if r.workload.startswith("long")], key=lambda x: x.warm_hit_ratio)
    short_pts = sorted([r for r in mixed_rows if r.workload.startswith("short")], key=lambda x: x.warm_hit_ratio)

    width = 960
    height = 540
    margin_l = 90
    margin_r = 40
    margin_t = 60
    margin_b = 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def xmap(hit: float) -> float:
        return margin_l + (hit - 0.5) / (0.95 - 0.5) * plot_w

    def ymap(savings_pct: float) -> float:
        return margin_t + (100.0 - savings_pct) / 100.0 * plot_h

    x_ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    y_ticks = [0, 20, 40, 60, 80, 100]

    long_poly = " ".join(f"{xmap(p.warm_hit_ratio):.1f},{ymap(p.savings_pct):.1f}" for p in long_pts)
    short_poly = " ".join(f"{xmap(p.warm_hit_ratio):.1f},{ymap(p.savings_pct):.1f}" for p in short_pts)

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
    parts.append('<text x="90" y="32" font-size="22" font-family="Arial" fill="#111">Savings vs Warm-Hit Ratio</text>')
    parts.append('<text x="90" y="52" font-size="13" font-family="Arial" fill="#444">Measured anchor model from benchmark summary</text>')

    # Grid and axes
    parts.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#222" stroke-width="1.5"/>')
    parts.append(f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#222" stroke-width="1.5"/>')
    for y in y_ticks:
        yy = ymap(float(y))
        parts.append(f'<line x1="{margin_l}" y1="{yy:.1f}" x2="{margin_l + plot_w}" y2="{yy:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="{margin_l - 12}" y="{yy + 4:.1f}" text-anchor="end" font-size="12" font-family="Arial" fill="#555">{y}%</text>')
    for x in x_ticks:
        xx = xmap(float(x))
        label = f"{int(x * 100)}%" if x != 0.95 else "95%"
        parts.append(f'<line x1="{xx:.1f}" y1="{margin_t}" x2="{xx:.1f}" y2="{margin_t + plot_h}" stroke="#f0f0f0" stroke-width="1"/>')
        parts.append(f'<text x="{xx:.1f}" y="{margin_t + plot_h + 22}" text-anchor="middle" font-size="12" font-family="Arial" fill="#555">{label}</text>')

    # Series
    parts.append(f'<polyline points="{long_poly}" fill="none" stroke="#0b62ff" stroke-width="3"/>')
    parts.append(f'<polyline points="{short_poly}" fill="none" stroke="#ff6b00" stroke-width="3"/>')
    for p in long_pts:
        parts.append(f'<circle cx="{xmap(p.warm_hit_ratio):.1f}" cy="{ymap(p.savings_pct):.1f}" r="4.2" fill="#0b62ff"/>')
    for p in short_pts:
        parts.append(f'<circle cx="{xmap(p.warm_hit_ratio):.1f}" cy="{ymap(p.savings_pct):.1f}" r="4.2" fill="#ff6b00"/>')

    # Legend
    lx = width - 275
    ly = margin_t + 10
    parts.append(f'<rect x="{lx}" y="{ly}" width="240" height="62" fill="#fafafa" stroke="#ddd"/>')
    parts.append(f'<line x1="{lx + 16}" y1="{ly + 22}" x2="{lx + 52}" y2="{ly + 22}" stroke="#0b62ff" stroke-width="3"/>')
    parts.append(f'<text x="{lx + 60}" y="{ly + 26}" font-size="13" font-family="Arial" fill="#222">Long context (120k / 128)</text>')
    parts.append(f'<line x1="{lx + 16}" y1="{ly + 45}" x2="{lx + 52}" y2="{ly + 45}" stroke="#ff6b00" stroke-width="3"/>')
    parts.append(f'<text x="{lx + 60}" y="{ly + 49}" font-size="13" font-family="Arial" fill="#222">Short context (4k / 128)</text>')

    parts.append(f'<text x="{margin_l + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle" font-size="12" font-family="Arial" fill="#555">Warm-hit ratio</text>')
    parts.append(f'<text transform="translate(20,{margin_t + plot_h/2:.1f}) rotate(-90)" text-anchor="middle" font-size="12" font-family="Arial" fill="#555">Savings (%)</text>')
    parts.append("</svg>")

    path.write_text("\n".join(parts), encoding="utf-8")


def _draw_line_chart(c: canvas.Canvas, x: float, y: float, w: float, h: float, mixed_rows: List[MixedPoint]) -> None:
    long_pts = sorted([r for r in mixed_rows if r.workload.startswith("long")], key=lambda r: r.warm_hit_ratio)
    short_pts = sorted([r for r in mixed_rows if r.workload.startswith("short")], key=lambda r: r.warm_hit_ratio)

    def xmap(hit: float) -> float:
        return x + (hit - 0.5) / (0.95 - 0.5) * w

    def ymap(save: float) -> float:
        return y + (save / 100.0) * h

    # axes
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.0)
    c.line(x, y, x, y + h)
    c.line(x, y, x + w, y)

    # grid + ticks
    c.setFont("Helvetica", 8)
    c.setStrokeColor(colors.HexColor("#d9d9d9"))
    for yt in [0, 20, 40, 60, 80, 100]:
        yy = ymap(float(yt))
        c.line(x, yy, x + w, yy)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawRightString(x - 4, yy - 2, f"{yt}%")
    for xt in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        xx = xmap(float(xt))
        c.setStrokeColor(colors.HexColor("#efefef"))
        c.line(xx, y, xx, y + h)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawCentredString(xx, y - 12, f"{int(xt*100)}%")

    # series
    c.setStrokeColor(colors.HexColor("#0b62ff"))
    c.setLineWidth(2.2)
    for i in range(1, len(long_pts)):
        c.line(
            xmap(long_pts[i - 1].warm_hit_ratio),
            ymap(long_pts[i - 1].savings_pct),
            xmap(long_pts[i].warm_hit_ratio),
            ymap(long_pts[i].savings_pct),
        )
    c.setStrokeColor(colors.HexColor("#ff6b00"))
    c.setLineWidth(2.2)
    for i in range(1, len(short_pts)):
        c.line(
            xmap(short_pts[i - 1].warm_hit_ratio),
            ymap(short_pts[i - 1].savings_pct),
            xmap(short_pts[i].warm_hit_ratio),
            ymap(short_pts[i].savings_pct),
        )

    c.setFillColor(colors.HexColor("#0b62ff"))
    for p in long_pts:
        c.circle(xmap(p.warm_hit_ratio), ymap(p.savings_pct), 2.8, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#ff6b00"))
    for p in short_pts:
        c.circle(xmap(p.warm_hit_ratio), ymap(p.savings_pct), 2.8, fill=1, stroke=0)

    # labels
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, y + h + 14, "Performance Chart: Savings vs Warm-Hit Ratio")
    c.setFont("Helvetica", 8)
    c.drawString(x, y + h + 2, "Blue: long context (120k/128)   Orange: short context (4k/128)")


def _write_pdf(path: Path, summary: dict, mixed_rows: List[MixedPoint], value_prop: str, hardware_note: str) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=12)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, spaceAfter=8)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.5, leading=14)
    mono = ParagraphStyle("Mono", parent=body, fontName="Courier", fontSize=9.5, leading=12)

    long_s = summary["long"]
    short_s = summary["short"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    story = []
    story.append(Paragraph("Axropus Benchmark Brief", h1))
    story.append(Paragraph(f"Generated: {now}", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("One-sentence value proposition", h2))
    story.append(Paragraph(value_prop, body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Measured outcome summary", h2))
    story.append(Paragraph(f"Test environment: {hardware_note}", body))
    story.append(Spacer(1, 6))
    table_data = [
        ["Metric", "Long Context (120k/128)", "Short Context (4k/128)"],
        ["Runs", str(int(long_s["runs"])), str(int(short_s["runs"]))],
        ["Hit rate", _fmt_pct(float(long_s["hit_rate"]) * 100.0), _fmt_pct(float(short_s["hit_rate"]) * 100.0)],
        ["Cold total", _fmt_ms(float(long_s["cold_total_ms"])), _fmt_ms(float(short_s["cold_total_ms"]))],
        ["Warm avg total", _fmt_ms(float(long_s["warm_total_ms_avg"])), _fmt_ms(float(short_s["warm_total_ms_avg"]))],
        ["End-to-end reduction", _fmt_pct(float(long_s["e2e_savings_pct"])), _fmt_pct(float(short_s["e2e_savings_pct"]))],
    ]
    t = Table(table_data, colWidths=[2.2 * inch, 2.1 * inch, 2.1 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f3f8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d4d9e2")),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 14))
    story.append(Paragraph("Method (plain language)", h2))
    story.append(
        Paragraph(
            "The runtime stores the computed prefix state on first request. On matching requests, "
            "it restores that state and skips prefix compute. Decode still runs normally. "
            "Results are from repeated-query benchmark traffic, not random production replay.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Technical Explanation (No Buzzwords)", h1))
    tech_points = [
        "1. Normalize prompt template and deterministic settings.",
        "2. Compute a stable key for the reusable prefix section.",
        "3. Miss path: run full prefill + decode, then store prefix state.",
        "4. Hit path: restore saved prefix state, skip prefill, run decode only.",
        "5. Track hit/miss, skip ratio, total latency, and decode latency per run.",
        "6. Require deterministic generation settings for valid replay behavior.",
    ]
    for p in tech_points:
        story.append(Paragraph(p, body))
        story.append(Spacer(1, 2))
    story.append(Spacer(1, 10))
    story.append(Paragraph("What this means operationally", h2))
    story.append(
        Paragraph(
            "Long contexts benefit most because prefill dominates first-run latency. "
            "Short contexts still benefit, but less, because decode occupies a larger share.",
            body,
        )
    )
    story.append(Spacer(1, 12))
    story.append(Paragraph("Data references used in this brief", h2))
    story.append(Paragraph("• summary.json (long + short benchmark aggregates)", mono))
    story.append(Paragraph("• Raw_Metrics_817runs_128k.csv (per-run long context records)", mono))
    story.append(Paragraph("• Mixed_Workload_Model.csv (sensitivity by warm-hit ratio)", mono))

    story.append(PageBreak())
    story.append(Paragraph("Performance Chart", h1))
    story.append(
        Paragraph(
            "Chart shows modeled savings versus warm-hit ratio using measured cold/warm anchors.",
            body,
        )
    )
    story.append(Spacer(1, 6))
    from reportlab.platypus import Flowable

    class ChartFlowable(Flowable):
        def __init__(self, mixed: List[MixedPoint]):
            super().__init__()
            self.mixed = mixed
            self.width = 6.6 * inch
            self.height = 3.9 * inch

        def draw(self) -> None:
            _draw_line_chart(self.canv, 0, 0, self.width, self.height, self.mixed)

        def wrap(self, availWidth, availHeight):
            return self.width, self.height + 0.2 * inch

    story.append(ChartFlowable(mixed_rows))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Selected points", h2))
    sel_tbl = Table(
        [
            ["Warm-hit ratio", "Long-context savings", "Short-context savings"],
            ["50%", "47.65%", "17.31%"],
            ["80%", "76.24%", "27.69%"],
            ["90%", "85.77%", "31.15%"],
            ["95%", "90.54%", "32.88%"],
        ],
        colWidths=[2.0 * inch, 2.1 * inch, 2.1 * inch],
    )
    sel_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f3f8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d4d9e2")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(sel_tbl)

    story.append(PageBreak())
    story.append(Paragraph("Contact-Ready Summary", h1))
    story.append(Paragraph("Use this wording directly in outbound email.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Value proposition", h2))
    story.append(Paragraph(value_prop, body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("What is proven", h2))
    proven = [
        f"• {int(long_s['runs'])} long-context runs, {int(long_s['hits'])} warm hits, {_fmt_pct(float(long_s['hit_rate']) * 100.0)} hit rate",
        f"• {_fmt_pct(float(long_s['e2e_savings_pct']))} end-to-end reduction on replayed long context",
        f"• {_fmt_pct(float(short_s['e2e_savings_pct']))} reduction floor on short/decode-heavy workload",
    ]
    for line in proven:
        story.append(Paragraph(line, body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Scope note", h2))
    story.append(
        Paragraph(
            "This benchmark is replay-heavy and deterministic by design. "
            "For buyer diligence, run a chaos+mixed-traffic validation on target hardware.",
            body,
        )
    )

    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build outbound benchmark pack (PDF + chart + explainer + value prop).")
    parser.add_argument(
        "--input-dir",
        default="platform_data/reports/michael_gibson_20260226_220847",
        help="Directory containing summary.json, Mixed_Workload_Model.csv, and raw metrics csv",
    )
    parser.add_argument(
        "--output-root",
        default="platform_data/reports",
        help="Root directory where outbound pack folder will be created",
    )
    parser.add_argument("--tag", default="", help="Optional suffix tag for output directory")
    parser.add_argument(
        "--hardware-note",
        default="Local single-GPU native engine benchmark (consumer RTX class, GGUF Q4 8B model).",
        help="Hardware/environment note shown in PDF and explainer",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_root).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_dir = output_root / f"outbound_pack_{timestamp}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(input_dir / "summary.json")
    mixed_rows = _load_mixed(input_dir / "Mixed_Workload_Model.csv")
    value_prop = _build_value_prop(summary)

    _write_value_prop(out_dir / "value_prop.txt", summary)
    _write_technical_explanation(out_dir / "technical_explanation.md", summary, args.hardware_note)
    _write_performance_chart_svg(out_dir / "performance_chart.svg", mixed_rows)
    _write_pdf(out_dir / "benchmark_brief.pdf", summary, mixed_rows, value_prop, args.hardware_note)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "artifacts": [
            "benchmark_brief.pdf",
            "performance_chart.svg",
            "technical_explanation.md",
            "value_prop.txt",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(str(out_dir))


if __name__ == "__main__":
    main()
