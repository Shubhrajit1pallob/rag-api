-- 004: answer cache. A stored answer is served again for an identical question
-- (same filters, models and prompt) instead of calling Voyage and Claude again.
-- Run once, in the Supabase SQL editor, on a database that already has 001-003.
-- For a fresh database use db/schema.sql instead.
-- Run it BEFORE deploying the app version that uses the cache; the older app ignores
-- the table. Everything is re-runnable except the final "create policy" statement
-- (Postgres has no "create policy if not exists").

-- 1. The cache. `key` is a sha256 of the normalized request; the question text itself is
--    never stored.
create table if not exists public.answer_cache (
  key        text primary key,
  answer     text not null,
  sources    jsonb not null,
  created_at timestamptz not null default now()
);
alter table public.answer_cache enable row level security;

-- 2. Any change to the documents empties the cache, in the same transaction as the
--    change, so a cached answer can never outlive the content it was built from.
--    The trigger runs with the caller's privileges (not SECURITY DEFINER), which is why
--    rag_api is granted delete on the table below.
create or replace function public.clear_answer_cache() returns trigger
language plpgsql as $$
begin
  delete from public.answer_cache where true;
  return null;
end;
$$;

drop trigger if exists documents_clear_answer_cache on public.documents;
create trigger documents_clear_answer_cache
  after insert or update or delete on public.documents
  for each statement execute function public.clear_answer_cache();

-- 3. Role grants and policy for the existing rag_api role (uncommented: role exists).
--    `update` is needed because storing an answer is an upsert.
grant select, insert, update, delete on public.answer_cache to rag_api;
create policy rag_api_all on public.answer_cache for all to rag_api using (true) with check (true);
