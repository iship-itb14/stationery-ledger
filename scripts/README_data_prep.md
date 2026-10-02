# Data preparation (done on a laptop; raw files are multi-GB)
1. Download from https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 :
   raw/meta_categories/meta_Office_Products.jsonl, meta_Arts_Crafts_and_Sewing.jsonl,
   and the first ~400 MB of raw/review_categories/Office_Products.jsonl (curl -r 0-400000000).
2. Stream each meta file line by line; keep items whose title/categories match stationery keywords
   (pen, pencil, marker, highlighter, notebook, journal, washi, sketchbook, ...), drop printer/toner/binder/etc.,
   require price 0 to 500 and at least 20 ratings -> data/stationery_meta_small.csv (36,854 rows).
3. Keep reviews of those items (cap 60,000) -> data/stationery_reviews_small.csv.
`analysis.py` then classifies product types, parses pack sizes, and runs the statistics and models.
