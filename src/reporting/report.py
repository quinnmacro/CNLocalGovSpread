"""
Report generation: HTML, Excel, and structured data exports.

Generates professional reports combining analysis results,
charts, and summary statistics for QuinnMacro showcase.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.core.logging_config import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Multi-format report generator for spread analysis results.

    Supports HTML (with embedded Plotly), Excel (multi-sheet), and JSON.
    """

    def __init__(
        self,
        title: str = "CN Local Gov Spread Analysis",
        author: str = "QuinnMacro",
    ) -> None:
        self._title = title
        self._author = author
        self._sections: list[dict[str, Any]] = []
        self._charts: list[dict] = []

    def add_section(
        self,
        name: str,
        content: str,
        data: dict | None = None,
    ) -> "ReportGenerator":
        """Add a text section to the report."""
        self._sections.append({
            "name": name,
            "content": content,
            "data": data or {},
        })
        return self

    def add_chart(self, name: str, fig_json: str) -> "ReportGenerator":
        """Add a Plotly chart (as JSON string) to the report."""
        self._charts.append({"name": name, "fig": fig_json})
        return self

    def add_summary_table(
        self,
        name: str,
        df: pd.DataFrame,
    ) -> "ReportGenerator":
        """Add a summary table to the report."""
        self._sections.append({
            "name": name,
            "content": "",
            "data": {"table": df.to_dict(orient="records")},
        })
        return self

    def generate_html(
        self,
        output_path: Path | str | None = None,
    ) -> str:
        """
        Generate a self-contained HTML report with embedded Plotly charts.

        Returns the HTML string and optionally writes to file.
        """
        sections_html = []
        for sec in self._sections:
            sec_html = f"<h2>{sec['name']}</h2>"
            if sec["content"]:
                sec_html += f"<div class='content'>{sec['content']}</div>"
            if "table" in sec.get("data", {}):
                table_df = pd.DataFrame(sec["data"]["table"])
                sec_html += table_df.to_html(classes="data-table", index=True)
            sections_html.append(sec_html)

        charts_html = []
        chart_scripts = []
        for i, chart in enumerate(self._charts):
            div_id = f"chart_{i}"
            charts_html.append(f'<div id="{div_id}" class="chart-container"></div>')
            chart_scripts.append(
                f'Plotly.newPlot("{div_id}", {chart["fig"]}.data, '
                f'{chart["fig"]}.layout, {{responsive: true}});'
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self._title}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
    <style>
        :root {{
            --bg: #0f172a; --fg: #f8fafc; --accent: #3b82f6;
            --surface: #1e293b; --border: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg); color: var(--fg);
            line-height: 1.6; padding: 2rem; max-width: 1200px; margin: auto;
        }}
        h1 {{ color: var(--accent); margin-bottom: 0.5rem; font-size: 2rem; }}
        h2 {{ color: #e2e8f0; margin: 2rem 0 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
        .header {{ margin-bottom: 2rem; }}
        .meta {{ color: #94a3b8; font-size: 0.875rem; }}
        .content {{ color: #cbd5e1; margin: 0.5rem 0; white-space: pre-wrap; }}
        .chart-container {{ margin: 1.5rem 0; min-height: 400px; }}
        .data-table {{
            width: 100%; border-collapse: collapse; margin: 1rem 0;
            font-size: 0.875rem;
        }}
        .data-table th, .data-table td {{
            padding: 0.5rem 1rem; border: 1px solid var(--border); text-align: right;
        }}
        .data-table th {{ background: var(--surface); color: var(--accent); text-align: left; }}
        .data-table tr:nth-child(even) {{ background: rgba(30,41,59,0.5); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self._title}</h1>
        <div class="meta">Author: {self._author} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
    </div>
    {"".join(sections_html)}
    <h2>Charts</h2>
    {"".join(charts_html)}
    <script>
        {"".join(chart_scripts)}
    </script>
</body>
</html>"""

        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")
            logger.info("HTML report written to %s", output_path)

        return html

    def generate_excel(
        self,
        output_path: Path | str,
        data_sheets: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """
        Generate multi-sheet Excel report.

        Parameters
        ----------
        data_sheets : dict
            {sheet_name: DataFrame} to include.
        """
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ImportError("Install openpyxl: pip install openpyxl")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            # Summary sheet
            summary_data = []
            for sec in self._sections:
                summary_data.append({"Section": sec["name"], "Content": sec["content"][:200]})
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

            # Data sheets
            if data_sheets:
                for name, df in data_sheets.items():
                    sheet_name = name[:31]  # Excel limit
                    df.to_excel(writer, sheet_name=sheet_name)

        logger.info("Excel report written to %s", path)

    def generate_json(self, output_path: Path | str | None = None) -> str:
        """Generate JSON report for programmatic consumption."""

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj: Any) -> Any:
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        report = {
            "title": self._title,
            "author": self._author,
            "generated": datetime.now().isoformat(),
            "sections": self._sections,
            "chart_count": len(self._charts),
        }

        json_str = json.dumps(report, indent=2, cls=NumpyEncoder)

        if output_path:
            Path(output_path).write_text(json_str, encoding="utf-8")
            logger.info("JSON report written to %s", output_path)

        return json_str

    def clear(self) -> "ReportGenerator":
        """Reset all sections and charts."""
        self._sections.clear()
        self._charts.clear()
        return self
