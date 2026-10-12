-- 002: global daily query budget + case-insensitive tag filtering.
-- Run once, in the Supabase SQL editor, on a database that already has 001.
-- For a fresh database use db/schema.sql instead.
--
-- Unlike 001/schema.sql, the role section at the bottom is UNCOMMENTED on purpose:
-- the rag_api role already exists in the Supabase project, so only the new
-- table's grant and policy are needed. The policy statement is not re-runnable
-- (Postgres has no "create policy if not exists"); everything above it is.

-- 1. Daily query budget (one row per UTC day).
create table if not exists public.query_usage (
  day   date primary key,
  count integer not null default 0
);
alter table public.query_usage enable row level security;

-- 2. Case-insensitive tags: tag_keys = lower(tags). `tags` keeps the casing as sent.
alter table public.documents add column if not exists tag_keys text[] not null default '{}';
update public.documents set tag_keys = array(select lower(t) from unnest(tags) as t);
drop index if exists public.documents_tags_gin;
create index if not exists documents_tag_keys_gin on public.documents using gin (tag_keys);

-- 3. Role grants and policy for the existing rag_api role (uncommented: role exists).
grant select, insert, update on public.query_usage to rag_api;
create policy rag_api_all on public.query_usage for all to rag_api using (true) with check (true);
