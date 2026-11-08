-- Supabase/Postgres schema for the review service (UNTESTED against a database).
-- Enable Row Level Security and restrict access to the service role / verified reviewers.
create table reviewer_credentials (
  user_id uuid primary key references auth.users(id),
  role text not null check (role in ('doctor','dietitian','fitness_professional','pharmacist')),
  verified boolean not null default false,
  verified_by uuid, verified_at timestamptz, license_ref text
);
create table review_items (
  id text primary key,
  record jsonb not null,                 -- AI-generated record, provenance 'ai_generated'
  status text not null default 'pending' check (status in ('pending','approved','rejected','disputed')),
  needed int not null default 1,
  lease_holder uuid, lease_until timestamptz,
  final jsonb, created_at timestamptz default now()
);
create table review_decisions (
  id bigserial primary key,
  item_id text references review_items(id),
  reviewer_id uuid not null references reviewer_credentials(user_id),
  decision text not null check (decision in ('approve','reject','correct')),
  corrected_answer text, note text, at timestamptz default now(),
  unique (item_id, reviewer_id)          -- one review per reviewer per item
);
create table review_audit (               -- append-only
  id bigserial primary key, at timestamptz default now(),
  action text not null, reviewer uuid, item text, detail jsonb
);
alter table review_items enable row level security;
alter table review_decisions enable row level security;
alter table review_audit enable row level security;
