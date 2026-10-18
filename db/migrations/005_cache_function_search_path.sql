-- 005: pin the search_path of public.clear_answer_cache() (Supabase Security Advisor:
-- "Function Search Path Mutable").
-- Run once, in the Supabase SQL editor, on a database that already has 004.
-- For a fresh database use db/schema.sql instead.
-- Re-runnable. Harmless to v0.2.0, and it may run before or after the v0.3.0 deploy:
-- no app code depends on it. The function body already names public.answer_cache in
-- full, so an empty search_path changes nothing it does.

alter function public.clear_answer_cache() set search_path = '';
