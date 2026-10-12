-- 001: original schema (documents + document_chunks). Already applied to the
-- existing Supabase project. For a fresh database use db/schema.sql instead,
-- which is the full current schema (001 + 002).
-- The role section is COMMENTED OUT: it needs a real generated password.
create extension if not exists vector with schema extensions;

create table public.documents (
  document_id     text primary key,
  title           text not null,
  category        text not null,
  status          text not null,
  date            text not null,
  featured        boolean not null default false,
  tags            text[] not null default '{}',
  content_hash    text not null,
  embedding_model text not null,
  max_chunk_size  integer not null,
  updated_at      timestamptz not null default now()
);

create table public.document_chunks (
  document_id   text not null references public.documents (document_id) on delete cascade,
  chunk_index   integer not null,
  section_title text not null default '',
  content       text not null,
  embedding     extensions.vector(1024) not null,
  primary key (document_id, chunk_index)
);

create index documents_tags_gin on public.documents using gin (tags);
create index documents_category_idx on public.documents (category);
-- No vector (HNSW) index yet: at portfolio scale an exact scan is fast and,
-- unlike HNSW, never returns fewer rows than requested when filters are applied.

alter table public.documents enable row level security;
alter table public.document_chunks enable row level security;

-- App role (least privilege). Run separately, with a real generated password:
-- create role rag_api login password '<generated>';
-- alter role rag_api set search_path = public, extensions;
-- grant usage on schema public, extensions to rag_api;
-- grant select, insert, update, delete on public.documents, public.document_chunks to rag_api;
-- create policy rag_api_all on public.documents for all to rag_api using (true) with check (true);
-- create policy rag_api_all on public.document_chunks for all to rag_api using (true) with check (true);
