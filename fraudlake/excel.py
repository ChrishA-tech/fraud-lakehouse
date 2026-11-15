"""Excel version of the report (openpyxl) plus a Power Query script that rebuilds it from the CSV exports."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

POWER_QUERY = r'''// Power Query (Excel: Data > Get Data > From Other Sources > Blank Query > Advanced Editor).
// Set FolderPath to the folder holding the CSV exports. Loads daily KPIs, types the columns, and adds a
// 7-day rolling fraud value so the pivot and chart in the workbook refresh from new exports.
let
    FolderPath = "C:\fraud-lakehouse\results\csv\",
    Source = Csv.Document(File.Contents(FolderPath & "daily_kpis.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"day", Int64.Type}, {"txns", Int64.Type}, {"fraud_txns", Int64.Type},
        {"fraud_rate", type number}, {"value", type number}, {"fraud_value", type number}}),
    Sorted = Table.Sort(Typed, {{"day", Order.Ascending}}),
    Indexed = Table.AddIndexColumn(Sorted, "i", 0, 1, Int64.Type),
    Rolling = Table.AddColumn(Indexed, "fraud_value_7d_pq", each List.Sum(List.Range(Indexed[fraud_value],
        List.Max({0, [i] - 6}), List.Min({7, [i] + 1}))), type number),
    Result = Table.RemoveColumns(Rolling, {"i"})
in
    Result
'''


def write(tables: Dict[str, pd.DataFrame], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "csv").mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / "csv" / f"{name}.csv", index=False)
    (out_dir / "power_query_daily_kpis.m").write_text(POWER_QUERY)
    path = out_dir / "fraud_report.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in tables.items():
            df.head(100000).to_excel(xw, sheet_name=name[:31], index=False)
        ws = xw.book[list(tables)[0][:31]]
        from openpyxl.chart import LineChart, Reference
        if ws.max_row > 2:
            ch = LineChart()
            ch.title, ch.y_axis.title, ch.x_axis.title = "Fraud value, 7-day rolling", "GBP", "Day"
            hdr = [c.value for c in ws[1]]
            if "fraud_value_7d" in hdr:
                col = hdr.index("fraud_value_7d") + 1
                ch.add_data(Reference(ws, min_col=col, min_row=1, max_row=ws.max_row), titles_from_data=True)
                ch.set_categories(Reference(ws, min_col=1, min_row=2, max_row=ws.max_row))
                ws.add_chart(ch, "M2")
        for w in xw.book.worksheets:
            for col in w.columns:
                w.column_dimensions[col[0].column_letter].width = min(max(len(str(col[0].value or "")) + 2, 10), 40)
    return path
