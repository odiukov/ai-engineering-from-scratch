"""Входные данные для замера скорости."""

_frameworks = ["SOC 2", "ISO 27001", "ISO 42001", "GDPR", "HIPAA", "EU AI Act",
               "PCI DSS", "Colorado AI Act"]

_record = {
    "ts": "2026-08-07T10:00:00Z", "user": "u1", "tenant": "t1",
    "action": "call_model", "model": "claude", "model_version": "2026-05",
    "risk_tier": "limited", "prompt_hash": "abc123", "response_hash": "def456",
    "pii_redacted": True, "phi_redacted": True, "legal_basis": "contract",
    "decision_outcome": "approved", "appeal_channel": "cs@example.com",
    "human_review": False,
}

_records = [
    {"ts": "%04d-%02d-%02d" % (2022 + i % 5, 1 + i % 12, 1 + i % 28), "n": i}
    for i in range(20000)
]

BENCH = {
    "frameworks_for": ("access logging",),
    "required_frameworks": ("Global", "enterprise"),
    "required_controls": (_frameworks,),
    "coverage_gaps": ([], _frameworks),
    "required_log_fields": (_frameworks,),
    "record_is_complete": (_record, _frameworks),
    "retention_days": (_frameworks,),
    "records_to_purge": (_records, _frameworks, "2026-08-07"),
}
