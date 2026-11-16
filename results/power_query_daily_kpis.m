// Power Query (Excel: Data > Get Data > From Other Sources > Blank Query > Advanced Editor).
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
