-- ============================================================
-- Supabase Schema for Reddit Sentiment Analyzer
-- ============================================================
-- Run this in Supabase SQL Editor to set up the database

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing table if migrating again
DROP TABLE IF EXISTS reddit_posts CASCADE;

-- Create main posts table
CREATE TABLE reddit_posts (
    -- Primary key
    post_id TEXT PRIMARY KEY,

    -- Reddit metadata
    subreddit TEXT NOT NULL,
    title TEXT NOT NULL,
    selftext TEXT,
    author TEXT,
    created_utc TIMESTAMPTZ NOT NULL,
    score INTEGER,
    num_comments INTEGER,
    url TEXT,
    permalink TEXT,
    collected_at TIMESTAMPTZ NOT NULL,

    -- Sentiment analysis (VADER)
    sentiment_pos REAL,
    sentiment_neg REAL,
    sentiment_neu REAL,
    sentiment_compound REAL,
    sentiment_label TEXT CHECK (sentiment_label IN ('positive', 'negative', 'neutral')),

    -- Vector embeddings (384 dimensions for all-MiniLM-L6-v2)
    embedding vector(384)
);

-- ============================================================
-- Indexes for Performance
-- ============================================================

-- Metadata indexes
CREATE INDEX idx_subreddit ON reddit_posts(subreddit);
CREATE INDEX idx_created_utc ON reddit_posts(created_utc DESC);
CREATE INDEX idx_sentiment_label ON reddit_posts(sentiment_label);
CREATE INDEX idx_sentiment_compound ON reddit_posts(sentiment_compound DESC);
CREATE INDEX idx_collected_at ON reddit_posts(collected_at DESC);

-- Vector similarity indexes (CRITICAL for fast semantic search)
-- ivfflat = Inverted File with Flat compression (good for <1M vectors)
-- lists = 100 is optimal for ~100K vectors (adjust based on dataset size)

-- Cosine similarity index (primary method)
CREATE INDEX idx_embedding_cosine ON reddit_posts
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- L2 distance index (alternative method)
CREATE INDEX idx_embedding_l2 ON reddit_posts
USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

-- ============================================================
-- Functions
-- ============================================================

-- Function: Search for similar posts
CREATE OR REPLACE FUNCTION search_similar_posts(
    query_embedding vector(384),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 20,
    filter_subreddit text DEFAULT NULL,
    filter_sentiment text DEFAULT NULL,
    days_ago int DEFAULT 30
)
RETURNS TABLE (
    post_id text,
    title text,
    selftext text,
    subreddit text,
    author text,
    created_utc timestamptz,
    score integer,
    sentiment_label text,
    sentiment_compound real,
    permalink text,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.post_id,
        p.title,
        p.selftext,
        p.subreddit,
        p.author,
        p.created_utc,
        p.score,
        p.sentiment_label,
        p.sentiment_compound,
        p.permalink,
        1 - (p.embedding <=> query_embedding) as similarity
    FROM reddit_posts p
    WHERE
        p.embedding IS NOT NULL
        AND (filter_subreddit IS NULL OR p.subreddit = filter_subreddit)
        AND (filter_sentiment IS NULL OR p.sentiment_label = filter_sentiment)
        AND p.created_utc >= NOW() - (days_ago || ' days')::INTERVAL
        AND (1 - (p.embedding <=> query_embedding)) > match_threshold
    ORDER BY p.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function: Get database statistics
CREATE OR REPLACE FUNCTION get_database_stats()
RETURNS TABLE (
    total_posts bigint,
    posts_with_embeddings bigint,
    posts_with_sentiment bigint,
    avg_sentiment_compound real,
    earliest_post timestamptz,
    latest_post timestamptz
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::bigint as total_posts,
        COUNT(embedding)::bigint as posts_with_embeddings,
        COUNT(sentiment_label)::bigint as posts_with_sentiment,
        AVG(sentiment_compound)::real as avg_sentiment_compound,
        MIN(created_utc) as earliest_post,
        MAX(created_utc) as latest_post
    FROM reddit_posts;
END;
$$;

-- ============================================================
-- Permissions
-- ============================================================

-- Grant SELECT permissions to anon key (for public queries)
GRANT SELECT ON reddit_posts TO anon;
GRANT EXECUTE ON FUNCTION search_similar_posts TO anon;
GRANT EXECUTE ON FUNCTION get_database_stats TO anon;

-- Grant full permissions to service role (for admin operations)
GRANT ALL ON reddit_posts TO service_role;

-- ============================================================
-- Verification Queries
-- ============================================================

-- Run these to verify setup:

-- 1. Check table structure
-- \d reddit_posts

-- 2. Check indexes
-- SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'reddit_posts';

-- 3. Verify pgvector extension
-- SELECT * FROM pg_extension WHERE extname = 'vector';

-- 4. Test statistics function
-- SELECT * FROM get_database_stats();

-- 5. Count posts (after migration)
-- SELECT COUNT(*) FROM reddit_posts;

-- 6. Check embedding coverage
-- SELECT
--     COUNT(*) as total,
--     COUNT(embedding) as with_embeddings,
--     COUNT(embedding)::float / COUNT(*) * 100 as coverage_pct
-- FROM reddit_posts;

-- ============================================================
-- Notes
-- ============================================================
-- - Embedding dimension: 384 (for all-MiniLM-L6-v2)
-- - Similarity metric: Cosine similarity (1 - cosine_distance)
-- - Index type: ivfflat (optimal for ~100K vectors)
-- - Free tier limit: 500MB (can hold ~200-300K posts with embeddings)
