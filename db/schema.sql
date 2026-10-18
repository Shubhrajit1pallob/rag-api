-- rag-api schema for Postgres + pgvector (Supabase or self-hosted).
-- This file always describes the FULL CURRENT schema: use it for a fresh database.
-- An existing database is upgraded by running the files in db/migrations/ in order
-- (001 = the original schema, 002 = query budget + case-insensitive tags,
-- 003 = chunker_version column, 004 = answer cache, 005 = fixed search_path on the
-- cache-clearing function).
--
-- The role section at the bottom is COMMENTED OUT on purpose: it needs a real
-- generated password, and the role is created once, separately. (db/migrations/002
-- runs the query_usage grants/policy uncommented because that role already exists.)
create extension if not exists vector with schema extensions;

create table public.documents (
  document_id     text primary key,
  title           text not null,
  category        text not null,
  status          text not null,
  date            text not null,
  featured        boolean not null default false,
  tags            text[] not null default '{}',   -- as sent, for display
  tag_keys        text[] not null default '{}',   -- lower(tags), for case-insensitive filters
  content_hash    text not null,
  embedding_model text not null,
  max_chunk_size  integer not null,
  chunker_version integer not null default 1,   -- app.pipeline.chunker.CHUNKER_VERSION at ingest
  updated_at     timestamptz not null default now()
);

create table public.document_chunks (
  document_id   text not null references public.documents (document_id) on delete cascade,
  chunk_index   integer not null,
  section_title text not null default '',
  content       text not null,
  embedding     extensions.vector(1024) not null,
  primary key (document_id, chunk_index)
);

-- Global daily query budget (the hard spend cap). One row per UTC day.
create table public.query_usage (
  day   date primary key,
  count integer not null default 0
);

-- Answer cache. `key` is a sha256 of the normalized request (no question text stored).
create table public.answer_cache (
  key        text primary key,
  answer     text not null,
  sources    jsonb not null,
  created_at timestamptz not null default now()
);

-- Any change to documents empties the cache, in the same transaction as the change.
-- Runs with the caller's privileges, so rag_api needs delete on answer_cache (below).
create or replace function public.clear_answer_cache() returns trigger
language plpgsql
set search_path = ''
as $$
begin
  delete from public.answer_cache where true;
  return null;
end;
$$;

create trigger documents_clear_answer_cache
  after insert or update or delete on public.documents
  for each statement execute function public.clear_answer_cache();

create index documents_tag_keys_gin on public.documents using gin (tag_keys);
create index documents_category_idx on public.documents (category);
-- No vector (HNSW) index yet: at portfolio scale an exact scan is fast and,
-- unlike HNSW, never returns fewer rows than requested when filters are applied.

alter table public.documents enable row level security;
alter table public.document_chunks enable row level security;
alter table public.query_usage enable row level security;
alter table public.answer_cache enable row level security;

-- App role (least privilege). Run separately, with a real generated password:
-- create role rag_api login password '<generated>';
-- alter role rag_api set search_path = public, extensions;
-- grant usage on schema public, extensions to rag_api;
-- grant select, insert, update, delete on public.documents, public.document_chunks to rag_api;
-- grant select, insert, update on public.query_usage to rag_api;
-- grant select, insert, update, delete on public.answer_cache to rag_api;
-- create policy rag_api_all on public.documents for all to rag_api using (true) with check (true);
-- create policy rag_api_all on public.document_chunks for all to rag_api using (true) with check (true);
-- create policy rag_api_all on public.query_usage for all to rag_api using (true) with check (true);
-- create policy rag_api_all on public.answer_cache for all to rag_api using (true) with check (true);
