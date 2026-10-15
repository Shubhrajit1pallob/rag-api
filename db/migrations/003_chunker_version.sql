-- 003: record which chunker version built each document's chunks.
-- Run once, in the Supabase SQL editor, on a database that already has 001 and 002.
-- For a fresh database use db/schema.sql instead.
-- Re-runnable. Run it BEFORE deploying the app version that reads this column;
-- the older app ignores the column, so running v0.1.0 pods are unaffected.
--
-- Existing rows get version 1 (the original chunker). The current app writes a higher
-- version, so the next ingest of each existing document re-chunks and re-embeds it once.

alter table public.documents add column if not exists chunker_version integer not null default 1;
