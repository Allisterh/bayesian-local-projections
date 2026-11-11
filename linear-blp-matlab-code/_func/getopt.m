function val = getopt(opts, field, default)
%GETOPT  Return opts.(field) if present, else default.
%
%Small shared option-parsing helper used throughout both codebases (the
%_run.m scripts, the _replication.m scripts, both *_core.m samplers, and
%the replication driver/wrapper BLP_replication_run.m/
%run_one_replication_case.m) to read a case_opts/opts field with a
%fallback default. Lives here in _func/ (on the path via each top-level
%script's own addpath([pwd '/_func/']), or BLP_replication_run.m's own
%addpath for the driver and its wrapper) rather than duplicated as a
%local function in multiple files, so there is exactly one copy to
%maintain.
if isfield(opts, field)
    val = opts.(field);
else
    val = default;
end
end
