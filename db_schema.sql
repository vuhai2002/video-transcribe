-- ============================================================
-- DATABASE SCHEMA: Video Subtitle Search System
-- PostgreSQL 14+
-- ============================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;       -- Trigram index for fuzzy search
CREATE EXTENSION IF NOT EXISTS unaccent;      -- Remove Vietnamese diacritics for search

-- ============================================================
-- 1. VIDEOS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS videos (
    id              SERIAL PRIMARY KEY,
    video_name      VARCHAR(500) NOT NULL,         -- Original filename
    video_path      TEXT,                          -- Path on storage (local/B2/Drive)
    drive_url       TEXT,                          -- Google Drive URL  
    b2_url          TEXT,                          -- Backblaze B2 URL
    duration        FLOAT DEFAULT 0,               -- Duration in seconds
    full_text       TEXT,                          -- Full transcript for quick search
    language        VARCHAR(10) DEFAULT 'vi',
    model_used      VARCHAR(50),                   -- e.g., "large-v3"
    total_segments  INTEGER DEFAULT 0,
    processing_time FLOAT DEFAULT 0,               -- Seconds to transcribe
    transcribed_at  TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW(),
    
    -- Unique constraint on video name to prevent duplicates
    CONSTRAINT uq_video_name UNIQUE (video_name)
);

-- ============================================================
-- 2. SUBTITLE SEGMENTS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS subtitle_segments (
    id              SERIAL PRIMARY KEY,
    video_id        INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    segment_id      INTEGER NOT NULL,              -- Order within video (0, 1, 2, ...)
    start_time      FLOAT NOT NULL,                -- Start time in seconds
    end_time        FLOAT NOT NULL,                -- End time in seconds
    text            TEXT NOT NULL,                  -- Subtitle text
    created_at      TIMESTAMP DEFAULT NOW(),
    
    -- Each segment is unique per video
    CONSTRAINT uq_segment UNIQUE (video_id, segment_id)
);


-- ============================================================
-- 4. SEARCH INDEXES (Critical for performance!)
-- ============================================================

-- GIN index for full-text search on Vietnamese content
-- This is the MAIN search index
CREATE INDEX IF NOT EXISTS idx_segments_text_search 
    ON subtitle_segments USING GIN (to_tsvector('simple', text));

-- GIN index for full-text search on full transcript
CREATE INDEX IF NOT EXISTS idx_videos_fulltext_search 
    ON videos USING GIN (to_tsvector('simple', full_text));

-- Trigram index for fuzzy/partial matching (e.g., "Quang Trun" → "Quang Trung")
CREATE INDEX IF NOT EXISTS idx_segments_text_trgm 
    ON subtitle_segments USING GIN (text gin_trgm_ops);

-- B-tree indexes for sorting and filtering
CREATE INDEX IF NOT EXISTS idx_segments_video_id 
    ON subtitle_segments(video_id);

CREATE INDEX IF NOT EXISTS idx_segments_time 
    ON subtitle_segments(video_id, start_time);

CREATE INDEX IF NOT EXISTS idx_word_timestamps_segment 
    ON word_timestamps(segment_id);

-- ============================================================
-- 5. SEARCH FUNCTIONS
-- ============================================================

-- Function: Search subtitles by keyword
-- Returns matching videos with segments containing the keyword
CREATE OR REPLACE FUNCTION search_subtitles(
    search_query TEXT,
    result_limit INTEGER DEFAULT 50
)
RETURNS TABLE (
    video_id        INTEGER,
    video_name      VARCHAR(500),
    video_path      TEXT,
    drive_url       TEXT,
    duration        FLOAT,
    segment_id      INTEGER,
    start_time      FLOAT,
    end_time        FLOAT,
    subtitle_text   TEXT,
    relevance       FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        v.id AS video_id,
        v.video_name,
        v.video_path,
        v.drive_url,
        v.duration,
        ss.segment_id,
        ss.start_time,
        ss.end_time,
        ss.text AS subtitle_text,
        ts_rank(to_tsvector('simple', ss.text), plainto_tsquery('simple', search_query)) AS relevance
    FROM subtitle_segments ss
    JOIN videos v ON v.id = ss.video_id
    WHERE 
        ss.text ILIKE '%' || search_query || '%'
    ORDER BY 
        v.video_name, 
        ss.start_time
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Function: Get all matching timestamps for a video
CREATE OR REPLACE FUNCTION get_video_matches(
    p_video_id INTEGER,
    search_query TEXT
)
RETURNS TABLE (
    segment_id      INTEGER,
    start_time      FLOAT,
    end_time        FLOAT,
    subtitle_text   TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ss.segment_id,
        ss.start_time,
        ss.end_time,
        ss.text AS subtitle_text
    FROM subtitle_segments ss
    WHERE 
        ss.video_id = p_video_id
        AND ss.text ILIKE '%' || search_query || '%'
    ORDER BY ss.start_time;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- 6. UTILITY VIEWS
-- ============================================================

-- View: Video summary with segment count
CREATE OR REPLACE VIEW video_summary AS
SELECT 
    v.id,
    v.video_name,
    v.duration,
    v.total_segments,
    v.transcribed_at,
    COUNT(ss.id) AS actual_segments,
    ROUND(v.duration / 60.0, 1) AS duration_minutes
FROM videos v
LEFT JOIN subtitle_segments ss ON ss.video_id = v.id
GROUP BY v.id;

-- ============================================================
-- 7. IMPORT FUNCTION (Load JSON output from transcribe.py)
-- ============================================================

-- This function will be called from the import script
-- to load JSON transcription results into the database
CREATE OR REPLACE FUNCTION import_transcription(p_data JSONB)
RETURNS INTEGER AS $$
DECLARE
    v_id INTEGER;
    seg JSONB;
BEGIN
    -- Insert or update video
    INSERT INTO videos (
        video_name, video_path, full_text, duration, 
        language, model_used, total_segments, transcribed_at
    ) VALUES (
        p_data->>'video_name',
        p_data->>'video_path',
        p_data->>'full_text',
        (p_data->>'duration')::FLOAT,
        p_data->>'language',
        p_data->>'model_used',
        (p_data->>'total_segments')::INTEGER,
        (p_data->>'transcribed_at')::TIMESTAMP
    )
    ON CONFLICT (video_name) DO UPDATE SET
        full_text = EXCLUDED.full_text,
        duration = EXCLUDED.duration,
        model_used = EXCLUDED.model_used,
        total_segments = EXCLUDED.total_segments,
        transcribed_at = EXCLUDED.transcribed_at,
        updated_at = NOW()
    RETURNING id INTO v_id;
    
    -- Delete existing segments for re-import
    DELETE FROM subtitle_segments WHERE video_id = v_id;
    
    -- Insert segments
    FOR seg IN SELECT jsonb_array_elements(p_data->'segments')
    LOOP
        INSERT INTO subtitle_segments (video_id, segment_id, start_time, end_time, text)
        VALUES (
            v_id,
            (seg->>'segment_id')::INTEGER,
            (seg->>'start_time')::FLOAT,
            (seg->>'end_time')::FLOAT,
            seg->>'text'
        );
    END LOOP;
    
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;
