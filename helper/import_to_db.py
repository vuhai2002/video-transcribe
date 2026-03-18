#!/usr/bin/env python3
"""
Import JSON transcription files into PostgreSQL database.

Usage:
    python3 import_to_db.py --json-dir ./output --db-url postgresql://user:pass@host:5432/dbname
    python3 import_to_db.py --json-file ./output/video1.json --db-url postgresql://user:pass@host:5432/dbname
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import Json
except ImportError:
    print("Installing psycopg2...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "psycopg2-binary"], check=True)
    import psycopg2
    from psycopg2.extras import Json

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def connect_db(db_url):
    """Connect to PostgreSQL database."""
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    return conn


def import_single_json(conn, json_path):
    """Import a single JSON transcription file into the database."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cur = conn.cursor()
    
    try:
        # Insert video record
        cur.execute("""
            INSERT INTO videos (
                video_name, video_path, full_text, duration,
                language, model_used, total_segments, 
                processing_time, transcribed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (video_name) DO UPDATE SET
                full_text = EXCLUDED.full_text,
                duration = EXCLUDED.duration,
                model_used = EXCLUDED.model_used,
                total_segments = EXCLUDED.total_segments,
                processing_time = EXCLUDED.processing_time,
                transcribed_at = EXCLUDED.transcribed_at,
                updated_at = NOW()
            RETURNING id
        """, (
            data['video_name'],
            data.get('video_path', ''),
            data.get('full_text', ''),
            data.get('duration', 0),
            data.get('language', 'vi'),
            data.get('model_used', ''),
            data.get('total_segments', 0),
            data.get('processing_time_seconds', 0),
            data.get('transcribed_at', datetime.now().isoformat())
        ))
        
        video_id = cur.fetchone()[0]
        
        # Delete existing segments for this video (re-import safe)
        cur.execute("DELETE FROM subtitle_segments WHERE video_id = %s", (video_id,))
        
        # Insert segments
        segments = data.get('segments', [])
        for seg in segments:
            cur.execute("""
                INSERT INTO subtitle_segments (video_id, segment_id, start_time, end_time, text)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                video_id,
                seg['segment_id'],
                seg['start'],
                seg['end'],
                seg['text']
            ))
        
        conn.commit()
        logger.info(f"✅ Imported: {data['video_name']} ({len(segments)} segments)")
        return True
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Failed to import {json_path}: {e}")
        return False


def import_directory(conn, json_dir):
    """Import all JSON files from a directory."""
    json_files = sorted(Path(json_dir).glob("*.json"))
    
    # Skip checkpoint.json
    json_files = [f for f in json_files if f.name != "checkpoint.json"]
    
    logger.info(f"Found {len(json_files)} JSON files to import")
    
    success = 0
    failed = 0
    
    for idx, json_file in enumerate(json_files, 1):
        logger.info(f"[{idx}/{len(json_files)}] Importing: {json_file.name}")
        
        if import_single_json(conn, str(json_file)):
            success += 1
        else:
            failed += 1
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Import complete: ✅ {success} | ❌ {failed} | Total: {len(json_files)}")
    logger.info(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Import transcriptions to PostgreSQL")
    parser.add_argument("--db-url", type=str, required=True,
                        help="PostgreSQL connection URL")
    parser.add_argument("--json-dir", type=str, default=None,
                        help="Directory containing JSON files")
    parser.add_argument("--json-file", type=str, default=None,
                        help="Single JSON file to import")
    parser.add_argument("--init-db", action="store_true",
                        help="Initialize database schema first")
    
    args = parser.parse_args()
    
    if not args.json_dir and not args.json_file:
        parser.error("Either --json-dir or --json-file is required")
    
    conn = connect_db(args.db_url)
    logger.info("Connected to PostgreSQL")
    
    # Initialize schema if requested
    if args.init_db:
        schema_file = os.path.join(os.path.dirname(__file__), "db_schema.sql")
        if os.path.exists(schema_file):
            with open(schema_file, 'r') as f:
                cur = conn.cursor()
                cur.execute(f.read())
                conn.commit()
                logger.info("Database schema initialized")
        else:
            logger.warning(f"Schema file not found: {schema_file}")
    
    # Import
    if args.json_file:
        import_single_json(conn, args.json_file)
    elif args.json_dir:
        import_directory(conn, args.json_dir)
    
    conn.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
