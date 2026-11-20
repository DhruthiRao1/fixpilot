from fixpilot.pii import redact, redact_settings


def test_redacts_email_and_recovery_key():
    text = "Signed in as jane.doe@example.com key 123456-123456-123456-123456-123456-123456-123456-123456"
    out = redact(text)
    assert "jane.doe@example.com" not in out and "[EMAIL]" in out
    assert "123456-123456" not in out and "[RECOVERY_KEY]" in out


def test_redacts_line_wrapped_bios_uuid_and_serial():
    text = "UUID 12345678-90ab-cdef-1234-567890a bcdef\nSerial Number 5CD1234XYZ\nBIOS Revision F.33"
    out = redact(text)
    assert "5CD1234XYZ" not in out
    assert "567890a" not in out
    assert "F.33" in out  # non-sensitive values survive


def test_redact_settings_by_key():
    out = redact_settings({"Serial Number": "5CD1234XYZ", "Secure Boot": "Enabled"})
    assert out == {"Serial Number": "[REDACTED]", "Secure Boot": "Enabled"}
