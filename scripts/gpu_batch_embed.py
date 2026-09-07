#!/usr/bin/env python3
"""
GPU batch embedding for the VPS worker DB.

Loads entries with NULL embeddings from a local copy of magnitu.db,
computes E5 embeddings on the RTX GPU using the magnitu-v3 pipeline,
and writes them back to the same DB in sub-batches.

Usage:
    python scripts/gpu_batch_embed.py --db magnitu_vps.db
    python scripts/gpu_batch_embed.py --db magnitu_vps.db --batch-size 64 --limit 1000
"""
import argparse
import json
import os
import sys
import tempfile
import time
import sqlite3
from typing import List, Optional

# Ensure we can import the magnitu package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Create a temp config dir with use_gpu=True so _select_device() picks CUDA.
# We point DB_PATH to a dummy (we use --db explicitly), but the config must
# enable GPU and match the VPS worker's embedding settings.
_tmp_data_dir = tempfile.mkdtemp(prefix="magnitu_gpu_embed_")
_tmp_cfg = {
    "seismo_url": "https://seismo.live/index.php",
    "api_key": "dummy",
    "use_gpu": True,
    "transformer_model_name": "intfloat/multilingual-e5-base",
    "embedding_dim": 768,
    "embedding_max_tokens": 512,
    "embedding_content_cap": 3000,
    "embedding_legal_content_cap": 12000,
    "embedding_analytical_content_cap": 7000,
    "legal_signal_patterns": [],
    "embedding_stack_generation": "e5-v3",
}
with open(os.path.join(_tmp_data_dir, "magnitu_config.json"), "w") as _f:
    json.dump(_tmp_cfg, _f)
os.environ["MAGNITU_DATA_DIR"] = _tmp_data_dir

import numpy as np
import config as config_mod
config_mod.load_config()
from pipeline import embed_entries, release_embedder, embedding_to_bytes


def load_entries_needing_embeddings(db_path: str, limit: int = 0) -> List[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    sql = (
        "SELECT entry_type, entry_id, title, description, content, "
        "source_type, source_name, source_category, published_date, link, author "
        "FROM entries WHERE embedding IS NULL "
        "ORDER BY entry_type, entry_id"
    )
    if limit > 0:
        sql += f" LIMIT {limit}"
    rows = conn.execute(sql).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def write_embeddings_batch(db_path: str, entries: List[dict], embeddings: List[bytes]):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for entry, emb in zip(entries, embeddings):
        cur.execute(
            "UPDATE entries SET embedding = ? WHERE entry_type = ? AND entry_id = ?",
            (emb, entry["entry_type"], entry["entry_id"]),
        )
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="GPU batch embedding for magnitu worker DB")
    parser.add_argument("--db", required=True, help="Path to the SQLite DB")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of entries to embed per GPU batch (default 100)")
    parser.add_argument("--write-batch", type=int, default=500,
                        help="Number of entries to write to DB per write batch (default 500)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max entries to process (0 = all)")
    args = parser.parse_args()

    db_path = args.db
    if not os.path.exists(db_path):
        print(f"ERROR: DB not found: {db_path}")
        sys.exit(1)

    entries = load_entries_needing_embeddings(db_path, limit=args.limit)
    n = len(entries)
    if n == 0:
        print("No entries need embedding — all done.")
        return

    print(f"Loaded {n} entries needing embeddings from {db_path}")
    print(f"GPU batch size: {args.batch_size}, write batch: {args.write_batch}")

    total_start = time.time()
    processed = 0
    write_buffer_entries = []
    write_buffer_embeddings = []

    for i in range(0, n, args.batch_size):
        batch = entries[i:i + args.batch_size]
        batch_start = time.time()

        embeddings = embed_entries(batch)

        write_buffer_entries.extend(batch)
        write_buffer_embeddings.extend(embeddings)

        batch_elapsed = time.time() - batch_start
        processed += len(batch)
        rate = len(batch) / batch_elapsed if batch_elapsed > 0 else 0
        eta = (n - processed) / rate if rate > 0 else 0

        print(f"  [{processed}/{n}] {len(batch)} entries in {batch_elapsed:.1f}s "
              f"({rate:.1f}/s, ETA {eta/60:.1f}min)")

        if len(write_buffer_entries) >= args.write_batch:
            write_embeddings_batch(db_path, write_buffer_entries, write_buffer_embeddings)
            print(f"    wrote {len(write_buffer_entries)} embeddings to DB")
            write_buffer_entries = []
            write_buffer_embeddings = []

    # Write remaining
    if write_buffer_entries:
        write_embeddings_batch(db_path, write_buffer_entries, write_buffer_embeddings)
        print(f"    wrote {len(write_buffer_entries)} embeddings to DB (final)")

    release_embedder()

    total_elapsed = time.time() - total_start
    print(f"\nDone: {n} entries embedded in {total_elapsed/60:.1f}min "
          f"({n/total_elapsed:.1f}/s avg)")

    # Verify
    conn = sqlite3.connect(db_path)
    with_emb = conn.execute("SELECT COUNT(*) FROM entries WHERE embedding IS NOT NULL").fetchone()[0]
    no_emb = conn.execute("SELECT COUNT(*) FROM entries WHERE embedding IS NULL").fetchone()[0]
    conn.close()
    print(f"DB state: {with_emb} with embeddings, {no_emb} without")


if __name__ == "__main__":
    main()
