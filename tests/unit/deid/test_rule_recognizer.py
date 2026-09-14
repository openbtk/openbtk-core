"""Unit tests for openbtk.deid.recognizers.rule.RuleRecognizer.

The contract suite proves generic behaviour (bounds, confidence, method
tagging); this file proves the actual detection logic -- one category at a
time -- and the phone/fax disambiguation, which is the one place this
recognizer needs more than a single independent regex.
"""

from __future__ import annotations

import pytest

from openbtk.deid.recognizers.rule import RuleRecognizer
from openbtk.deid.schemas import PHICategory

# Named separately (rather than inline in _CASES below) so each synthetic
# value that matches tests/security/test_fixture_hygiene.py's scanner
# patterns (docs/06_SECURITY_COMPLIANCE.md section 3.2) can carry its own
# single-line "phi-fixture-ok" marker -- the scanner requires the marker on
# the exact line containing the match.
_SSN_VALUE = "123-45-6789"  # phi-fixture-ok
_EMAIL_VALUE = "jane.doe@example.org"  # phi-fixture-ok
_MRN_VALUE = "MRN-4821093"  # phi-fixture-ok

_CASES: list[tuple[str, PHICategory, str]] = [
    (f"SSN: {_SSN_VALUE}", PHICategory.SSN, _SSN_VALUE),
    (f"Email: {_EMAIL_VALUE}", PHICategory.EMAIL, _EMAIL_VALUE),
    (
        "Portal: https://patient-portal.example.com/x",
        PHICategory.URL,
        "https://patient-portal.example.com/x",
    ),
    ("Last login IP: 192.168.1.42", PHICategory.IP_ADDRESS, "192.168.1.42"),
    ("DOB: 03/14/1958", PHICategory.DATE, "03/14/1958"),
    (f"MRN: {_MRN_VALUE}", PHICategory.MEDICAL_RECORD_NUMBER, _MRN_VALUE),
    (
        "Health Plan ID: AB123456",
        PHICategory.HEALTH_PLAN_BENEFICIARY_NUMBER,
        "AB123456",
    ),
    ("License: LIC-4F9K2A", PHICategory.CERTIFICATE_LICENSE_NUMBER, "LIC-4F9K2A"),
    ("Vehicle: 7GK-4821", PHICategory.VEHICLE_IDENTIFIER, "7GK-4821"),
    ("Device SN: SN-4F9K2A7B", PHICategory.DEVICE_IDENTIFIER, "SN-4F9K2A7B"),
    ("Reference ID: ID-4f9b2a71", PHICategory.OTHER_UNIQUE_IDENTIFIER, "ID-4f9b2a71"),
    ("Account #: 5849302716", PHICategory.ACCOUNT_NUMBER, "5849302716"),
]


class TestPerCategoryDetection:
    @pytest.mark.parametrize(("text", "category", "value"), _CASES)
    def test_detects_the_expected_category_and_value(
        self, text: str, category: PHICategory, value: str
    ) -> None:
        detections = RuleRecognizer().detect(text)
        matches = [d for d in detections if d.category == category]
        assert len(matches) == 1, f"expected exactly one {category} match in {text!r}"
        detection = matches[0]
        assert text[detection.span.start : detection.span.end] == value


class TestPhoneFaxDisambiguation:
    def test_phone_labelled_text_is_phone_number(self) -> None:
        detections = RuleRecognizer().detect("Phone: (555) 234-5678")
        assert detections[0].category == PHICategory.PHONE_NUMBER

    def test_fax_labelled_text_is_fax_number(self) -> None:
        detections = RuleRecognizer().detect("Fax: (555) 234-5678")
        assert detections[0].category == PHICategory.FAX_NUMBER

    def test_fax_keyword_outside_the_window_does_not_affect_classification(
        self,
    ) -> None:
        # "Fax" appears, but far enough before the number that it belongs to
        # a different field -- the window must not reach back that far.
        text = "Fax machine broken. Contact main desk at (555) 234-5678 instead."
        detections = RuleRecognizer().detect(text)
        assert detections[0].category == PHICategory.PHONE_NUMBER


class TestDoesNotAttemptNerCategories:
    """Documents, behaviourally, the module docstring's explicit scope
    limitation -- a regression guard against someone later bolting a naive
    name/address regex onto this recognizer and quietly violating the
    "no accuracy claim without a benchmark" rule."""

    def test_a_name_alone_produces_no_detections(self) -> None:
        assert RuleRecognizer().detect("Jane Quinn Patient") == []

    def test_a_street_address_alone_produces_no_detections(self) -> None:
        assert RuleRecognizer().detect("742 Evergreen Terrace") == []


class TestAgainstTheLabelledCorpusTemplate:
    def test_detects_every_rule_coverable_category_in_one_document(self) -> None:
        """Integration-style check against a realistic, multi-category
        document shaped like tests/fixtures/labelled_phi_corpus.py's own
        template -- not the corpus itself (that's tests/accuracy/'s job),
        just confirming nothing collides when categories appear together."""
        text = (
            "Patient: Jane Q. Patient\n"
            "DOB: 03/14/1958\n"
            "MRN: MRN-4821093  Account #: 5849302716\n"  # phi-fixture-ok
            "Address: 742 Evergreen Terrace\n"
            "Phone: (555) 234-5678  Fax: (555) 234-9999  "
            "Email: jane.patient@example.org\n"  # phi-fixture-ok
            "SSN: 123-45-6789  Health Plan ID: AB123456\n"  # phi-fixture-ok
            "Vehicle: 7GK-4821  Device SN: SN-4F9K2A7B\n"
            "License: LIC-4F9K2A\n"
            "Portal: https://portal.example.com  Last login IP: 192.168.1.42\n"
            "Reference ID: ID-4f9b2a71\n"
        )
        detections = RuleRecognizer().detect(text)
        found_categories = {d.category for d in detections}
        expected = {
            PHICategory.DATE,
            PHICategory.MEDICAL_RECORD_NUMBER,
            PHICategory.ACCOUNT_NUMBER,
            PHICategory.PHONE_NUMBER,
            PHICategory.FAX_NUMBER,
            PHICategory.EMAIL,
            PHICategory.SSN,
            PHICategory.HEALTH_PLAN_BENEFICIARY_NUMBER,
            PHICategory.VEHICLE_IDENTIFIER,
            PHICategory.DEVICE_IDENTIFIER,
            PHICategory.CERTIFICATE_LICENSE_NUMBER,
            PHICategory.URL,
            PHICategory.IP_ADDRESS,
            PHICategory.OTHER_UNIQUE_IDENTIFIER,
        }
        assert found_categories == expected
