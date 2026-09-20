"""``HL7v2Loader`` (FR-E-03). Every message is hand-written and synthetic; the names,
addresses and numbers are made up (``# phi-fixture-ok``)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from openbtk.core.errors import LoaderError
from openbtk.core.registry import LOADER_REGISTRY
from openbtk.core.schemas import CodeSystem
from openbtk.data.ehr import hl7v2
from openbtk.data.ehr.hl7v2 import HL7v2Loader

if TYPE_CHECKING:
    from pathlib import Path

    from openbtk.data.ehr.schemas import PatientRecord

pytest.importorskip("hl7apy")

_MSH = "MSH|^~\\&|APP|FAC|RCV|FAC|20240314101500+0000||{type}|MSG{n}|P|2.5"


def _message(kind: str, *segments: str, n: int = 1) -> str:
    return "\r".join([_MSH.format(type=kind, n=n), *segments])


def _pid(
    ident: str = "PT0001^^^HOSP^MR",
    *,
    sex: str = "F",
    born: str = "19800601",
    **more: str,
) -> str:
    extra = more.get("extra", "")
    return f"PID|1||{ident}||DOE^JANE||{born}|{sex}{extra}"


def _write(tmp_path: Path, *messages: str, name: str = "a.hl7") -> str:
    (tmp_path / name).write_text("\r".join(messages), encoding="utf-8")
    return str(tmp_path)


def _load(source: str, **kwargs: Any) -> list[PatientRecord]:
    return list(HL7v2Loader(**kwargs).load(source))


def test_is_registered() -> None:
    assert "loader.ehr.hl7v2" in LOADER_REGISTRY.list_keys()


class TestDemographics:
    def test_the_basics(self, tmp_path: Path) -> None:
        source = _write(tmp_path, _message("ADT^A01", _pid()))
        (record,) = _load(source)
        assert record.patient_id == "PT0001"
        assert record.demographics.birth_date == date(1980, 6, 1)
        assert record.demographics.gender == "female"
        assert record.source_system == "hl7v2"

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("M", "male"),
            ("F", "female"),
            ("O", "other"),
            ("A", "other"),
            ("X", "other"),
            ("U", "unknown"),
            ("N", "unknown"),
            ("m", "male"),
        ],
    )
    def test_administrative_sex_maps_through_hl7_table_0001(
        self, tmp_path: Path, code: str, expected: str
    ) -> None:
        (record,) = _load(_write(tmp_path, _message("ADT^A01", _pid(sex=code))))
        assert record.demographics.gender == expected

    def test_an_unknown_sex_code_is_left_unset_not_guessed(
        self, tmp_path: Path
    ) -> None:
        (record,) = _load(_write(tmp_path, _message("ADT^A01", _pid(sex="Q"))))
        assert record.demographics.gender is None

    def test_race_ethnicity_and_death(self, tmp_path: Path) -> None:
        # PID-10 race, PID-22 ethnic group, PID-29 death time, PID-30 death indicator
        cells = [
            "PID",
            "1",
            "",
            "PT0001^^^HOSP^MR",
            "",
            "DOE^JANE",
            "",
            "19800601",
            "F",
        ]
        cells += [""] * (10 - len(cells) + 1)  # up to PID-10
        cells[10] = "2106-3^White^HL70005"
        cells += [""] * (22 - len(cells) + 1)
        cells[22] = "N^Not Hispanic^HL70189"
        cells += [""] * (29 - len(cells) + 1)
        cells[29] = "20240201"
        cells += ["Y"]
        (record,) = _load(_write(tmp_path, _message("ADT^A03", "|".join(cells))))
        assert record.demographics.race == "White"
        assert record.demographics.ethnicity == "Not Hispanic"
        assert record.demographics.deceased is True
        assert record.demographics.deceased_date == date(2024, 2, 1)

    def test_the_mr_identifier_is_preferred_among_repeats(self, tmp_path: Path) -> None:
        ident = "SSN9^^^USA^SS~MRN7^^^HOSP^MR"
        (record,) = _load(_write(tmp_path, _message("ADT^A01", _pid(ident))))
        assert record.patient_id == "MRN7"

    def test_without_an_mr_the_first_identifier_is_used(self, tmp_path: Path) -> None:
        (record,) = _load(_write(tmp_path, _message("ADT^A01", _pid("ABC^^^X"))))
        assert record.patient_id == "ABC"

    def test_a_message_without_a_patient_id_is_ignored(self, tmp_path: Path) -> None:
        source = _write(
            tmp_path,
            _message("ADT^A01", "PID|1||||DOE^JANE"),
            _message("ADT^A01", _pid(), n=2),
        )
        assert [r.patient_id for r in _load(source)] == ["PT0001"]

    def test_a_message_without_pid_is_ignored(self, tmp_path: Path) -> None:
        source = _write(tmp_path, _message("QRY^A19", "QRD|20240314"))
        assert _load(source) == []


class TestNothingBeyondWhatIsNeededIsRead:
    def test_names_addresses_and_phones_never_reach_the_record(
        self, tmp_path: Path
    ) -> None:
        cells = [
            "PID",
            "1",
            "",
            "PT0001^^^HOSP^MR",
            "",
            "Zebediah^Quuxington",
            "",
            "19800601",
            "F",
        ]
        cells += [""] * (11 - len(cells) + 1)
        cells[11] = "742 Evergreen Sentinel Terrace^^Springfield^ZZ^00000"
        cells += [""] * (13 - len(cells) + 1)
        cells[13] = "555-0100-SENTINEL"
        (record,) = _load(_write(tmp_path, _message("ADT^A01", "|".join(cells))))
        dumped = record.model_dump_json()
        for sentinel in (
            "Zebediah",
            "Quuxington",
            "Evergreen",
            "Springfield",
            "555-0100",
        ):
            assert sentinel not in dumped


class TestEncounters:
    def _pv1(
        self, klass: str = "I", visit: str = "VN1", start: str = "", end: str = ""
    ) -> str:
        cells = ["PV1", "1", klass] + [""] * 17
        cells[19] = visit
        cells += [""] * (45 - len(cells) + 1)
        cells[44], cells[45] = start, end
        return "|".join(cells)

    def test_an_admission(self, tmp_path: Path) -> None:
        source = _write(
            tmp_path, _message("ADT^A01", _pid(), self._pv1(start="20240314101500"))
        )
        (record,) = _load(source)
        (encounter,) = record.encounters
        assert encounter.encounter_id == "VN1"
        assert encounter.encounter_type == "inpatient"
        assert encounter.start == datetime(2024, 3, 14, 10, 15, tzinfo=UTC)
        assert encounter.status == "in-progress"

    def test_a_later_discharge_updates_the_same_encounter(self, tmp_path: Path) -> None:
        source = _write(
            tmp_path,
            _message("ADT^A01", _pid(), self._pv1(start="20240314101500"), n=1),
            _message("ADT^A03", _pid(), self._pv1(end="20240316090000"), n=2),
        )
        (record,) = _load(source)
        (encounter,) = record.encounters
        assert encounter.start is not None and encounter.end is not None
        assert encounter.status == "finished"
        assert encounter.encounter_type == "inpatient"

    @pytest.mark.parametrize(
        ("klass", "expected"),
        [("O", "outpatient"), ("E", "emergency"), ("P", "preadmit"), ("Z", None)],
    )
    def test_patient_class_maps_through_table_0004(
        self, tmp_path: Path, klass: str, expected: str | None
    ) -> None:
        source = _write(tmp_path, _message("ADT^A01", _pid(), self._pv1(klass=klass)))
        (record,) = _load(source)
        assert record.encounters[0].encounter_type == expected

    def test_a_visit_without_a_number_makes_no_encounter(self, tmp_path: Path) -> None:
        source = _write(tmp_path, _message("ADT^A01", _pid(), "PV1|1|I"))
        (record,) = _load(source)
        assert record.encounters == []

    def test_events_link_to_the_visit_of_their_message(self, tmp_path: Path) -> None:
        source = _write(
            tmp_path,
            _message(
                "ADT^A01",
                _pid(),
                self._pv1(),
                "DG1|1||I10^Essential hypertension^I10C||20240314",
            ),
        )
        (record,) = _load(source)
        assert record.conditions[0].encounter_ref == "VN1"


class TestCodedEvents:
    def test_diagnosis_procedure_and_medications(self, tmp_path: Path) -> None:
        rxa = "RXA|0|1|20240314090000||197361^Amlodipine 5 MG^RXNORM|5|mg"
        source = _write(
            tmp_path,
            _message(
                "ADT^A01",
                _pid(),
                "DG1|1||I10^Essential hypertension^I10C||20240314",
                "PR1|1||99213^Office visit^C4||20240314",
                rxa,
                "RXE|1|860975^Metformin 500 MG^RXNORM",
            ),
        )
        (record,) = _load(source)
        (dx,) = record.conditions
        assert (dx.code, dx.system, dx.display) == (
            "I10",
            CodeSystem.ICD10CM,
            "Essential hypertension",
        )
        assert record.procedures[0].system is CodeSystem.CPT
        assert {m.code for m in record.medications} == {"197361", "860975"}
        assert all(m.system is CodeSystem.RXNORM for m in record.medications)
        assert record.medications[0].timestamp == datetime(2024, 3, 14, 9, tzinfo=UTC)

    def test_plain_i10_is_who_icd_10_and_is_not_mapped_to_icd_10_cm(
        self, tmp_path: Path
    ) -> None:
        source = _write(
            tmp_path,
            _message("ADT^A01", _pid(), "DG1|1||I10^Hypertension^I10||20240314"),
        )
        (record,) = _load(source)
        assert record.conditions == []

    def test_the_alternate_coding_is_tried_when_the_primary_is_unmappable(
        self, tmp_path: Path
    ) -> None:
        dg1 = "DG1|1||LOCAL1^Local^L^38341003^Hypertensive disorder^SCT||20240314"
        (record,) = _load(_write(tmp_path, _message("ADT^A01", _pid(), dg1)))
        assert (record.conditions[0].code, record.conditions[0].system) == (
            "38341003",
            CodeSystem.SNOMED,
        )

    def test_an_unmappable_event_is_skipped_counted_and_never_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = MagicMock()
        monkeypatch.setattr(hl7v2, "log", spy)
        dg1 = "DG1|1||X1^Unknown^ZZZ||20240314"
        (record,) = _load(_write(tmp_path, _message("ADT^A01", _pid(), dg1, dg1)))
        assert record.conditions == []
        assert spy.warning.call_args.args == ("loader.hl7v2.skipped",)
        assert spy.warning.call_args.kwargs["events"] == 2

    def test_a_repeated_event_is_kept_once(self, tmp_path: Path) -> None:
        dg1 = "DG1|1||I10^Essential hypertension^I10C||20240314"
        source = _write(
            tmp_path,
            _message("ADT^A01", _pid(), dg1, n=1),
            _message("ADT^A08", _pid(), dg1, n=2),
        )
        (record,) = _load(source)
        assert len(record.conditions) == 1


class TestObservations:
    def _obx(
        self,
        kind: str = "NM",
        ident: str = "718-7^Hemoglobin^LN",
        value: str = "13.5",
        units: str = "g/dL^g/dL^UCUM",
        low_high: str = "12-16",
        status: str = "F",
        when: str = "20240314103000",
    ) -> str:
        cells = [
            "OBX",
            "1",
            kind,
            ident,
            "",
            value,
            units,
            low_high,
            "N",
            "",
            "",
            status,
            "",
            "",
            when,
        ]
        return "|".join(cells)

    def _one(self, tmp_path: Path, obx: str) -> PatientRecord:
        (record,) = _load(_write(tmp_path, _message("ORU^R01", _pid(), obx)))
        return record

    def test_a_numeric_result(self, tmp_path: Path) -> None:
        (m,) = self._one(tmp_path, self._obx()).observations
        assert (m.code, m.system, m.display) == (
            "718-7",
            CodeSystem.LOINC,
            "Hemoglobin",
        )
        assert (
            m.value == 13.5 and m.unit == "g/dL" and m.reference_range == (12.0, 16.0)
        )
        assert m.timestamp == datetime(2024, 3, 14, 10, 30, tzinfo=UTC)

    def test_a_text_result(self, tmp_path: Path) -> None:
        (m,) = self._one(
            tmp_path, self._obx(kind="ST", value="Positive", units="")
        ).observations
        assert m.value == "Positive" and m.unit is None

    def test_a_coded_result_uses_its_text(self, tmp_path: Path) -> None:
        obx = self._obx(kind="CWE", value="POS^Positive^L", units="")
        assert self._one(tmp_path, obx).observations[0].value == "Positive"

    @pytest.mark.parametrize("status", ["D", "X"])
    def test_deleted_and_not_performed_results_are_skipped(
        self, tmp_path: Path, status: str
    ) -> None:
        assert self._one(tmp_path, self._obx(status=status)).observations == []

    def test_a_non_numeric_nm_is_skipped_not_guessed(self, tmp_path: Path) -> None:
        assert self._one(tmp_path, self._obx(value="high")).observations == []

    def test_an_unsupported_value_type_is_skipped(self, tmp_path: Path) -> None:
        assert (
            self._one(tmp_path, self._obx(kind="ED", value="binary")).observations == []
        )

    def test_a_non_ucum_unit_is_not_passed_off_as_ucum(self, tmp_path: Path) -> None:
        obx = self._obx(units="g/dl^gram per dl^ISO+")
        assert self._one(tmp_path, obx).observations[0].unit is None

    def test_a_unit_without_a_system_is_taken_as_written(self, tmp_path: Path) -> None:
        assert (
            self._one(tmp_path, self._obx(units="mg/dL")).observations[0].unit
            == "mg/dL"
        )

    def test_a_reference_range_that_is_not_low_high_is_dropped(
        self, tmp_path: Path
    ) -> None:
        assert (
            self._one(tmp_path, self._obx(low_high="<5"))
            .observations[0]
            .reference_range
            is None
        )

    def test_an_unmapped_observation_code_is_skipped(self, tmp_path: Path) -> None:
        obx = self._obx(ident="X1^Local^99LOCAL")
        assert self._one(tmp_path, obx).observations == []

    def test_escape_sequences_are_resolved(self, tmp_path: Path) -> None:
        obx = self._obx(kind="ST", value="a\\F\\b\\S\\c", units="")
        assert self._one(tmp_path, obx).observations[0].value == "a|b^c"


class TestTimestamps:
    def _when(self, tmp_path: Path, value: str, **kwargs: Any) -> datetime | None:
        obx = f"OBX|1|NM|718-7^Hb^LN||1||||||F|||{value}"
        (record,) = _load(_write(tmp_path, _message("ORU^R01", _pid(), obx)), **kwargs)
        return record.observations[0].timestamp

    def test_an_explicit_offset_is_honoured(self, tmp_path: Path) -> None:
        got = self._when(tmp_path, "20240314103000-0500")
        assert got == datetime(
            2024, 3, 14, 10, 30, tzinfo=timezone(timedelta(hours=-5))
        )

    def test_no_offset_means_utc_by_default(self, tmp_path: Path) -> None:
        assert self._when(tmp_path, "20240314103000") == datetime(
            2024, 3, 14, 10, 30, tzinfo=UTC
        )

    def test_no_offset_uses_the_configured_default(self, tmp_path: Path) -> None:
        got = self._when(tmp_path, "20240314103000", default_utc_offset_hours=2)
        assert got is not None and got.utcoffset() == timedelta(hours=2)

    def test_reduced_precision_is_accepted(self, tmp_path: Path) -> None:
        assert self._when(tmp_path, "202403") == datetime(2024, 3, 1, tzinfo=UTC)

    def test_fractional_seconds_are_accepted(self, tmp_path: Path) -> None:
        assert self._when(tmp_path, "20240314103000.25") == datetime(
            2024, 3, 14, 10, 30, tzinfo=UTC
        )

    @pytest.mark.parametrize("bad", ["notadate", "20241399", "2024-03-14"])
    def test_an_unparseable_time_falls_back_to_the_message_time(
        self, tmp_path: Path, bad: str
    ) -> None:
        assert self._when(tmp_path, bad) == datetime(2024, 3, 14, 10, 15, tzinfo=UTC)


class TestFilesAndFraming:
    def test_several_patients_in_one_file_come_out_in_order(
        self, tmp_path: Path
    ) -> None:
        source = _write(
            tmp_path,
            _message("ADT^A01", _pid("B^^^H^MR"), n=1),
            _message("ADT^A01", _pid("A^^^H^MR"), n=2),
            _message("ADT^A08", _pid("B^^^H^MR"), n=3),
        )
        assert [r.patient_id for r in _load(source)] == ["B", "A"]

    def test_the_same_patient_in_two_files_is_two_records(self, tmp_path: Path) -> None:
        _write(tmp_path, _message("ADT^A01", _pid()), name="a.hl7")
        _write(tmp_path, _message("ADT^A01", _pid()), name="b.hl7")
        assert len(_load(str(tmp_path))) == 2

    def test_a_single_file_can_be_the_source(self, tmp_path: Path) -> None:
        _write(tmp_path, _message("ADT^A01", _pid()))
        assert len(_load(str(tmp_path / "a.hl7"))) == 1

    def test_only_hl7_files_in_a_directory_are_read(self, tmp_path: Path) -> None:
        _write(tmp_path, _message("ADT^A01", _pid()), name="a.hl7")
        (tmp_path / "notes.txt").write_text("not a message", encoding="utf-8")
        assert len(_load(str(tmp_path))) == 1

    def test_lf_crlf_and_mllp_framing_are_tolerated(self, tmp_path: Path) -> None:
        message = _message("ADT^A01", _pid()).replace("\r", "\r\n")
        (tmp_path / "a.hl7").write_text(f"\x0b{message}\x1c\r", encoding="utf-8")
        (tmp_path / "b.hl7").write_text(message.replace("\r\n", "\n"), encoding="utf-8")
        assert len(_load(str(tmp_path))) == 2

    def test_batch_wrappers_are_skipped(self, tmp_path: Path) -> None:
        text = "\r".join(
            ["FHS|^~\\&", "BHS|^~\\&", _message("ADT^A01", _pid()), "BTS|1", "FTS|1"]
        )
        (tmp_path / "a.hl7").write_text(text, encoding="utf-8")
        assert len(_load(str(tmp_path))) == 1

    def test_it_reads_one_file_at_a_time(self, tmp_path: Path) -> None:
        _write(tmp_path, _message("ADT^A01", _pid()), name="a.hl7")
        (tmp_path / "b.hl7").write_text("this is not hl7", encoding="utf-8")
        records = HL7v2Loader().load(str(tmp_path))
        assert next(records).patient_id == "PT0001"  # before b.hl7 is even opened
        with pytest.raises(LoaderError, match=r"b\.hl7"):
            next(records)


class TestErrors:
    def test_a_missing_source(self, tmp_path: Path) -> None:
        with pytest.raises(LoaderError, match="Not a file or directory"):
            list(HL7v2Loader().load(str(tmp_path / "nope")))

    def test_an_unparseable_message_names_the_file_and_position_not_the_content(
        self, tmp_path: Path
    ) -> None:
        good = _message("ADT^A01", _pid())
        (tmp_path / "a.hl7").write_text(
            good + "\rMSH|" + "\r" + "SECRET-CONTENT", encoding="utf-8"
        )
        with pytest.raises(LoaderError) as excinfo:
            list(HL7v2Loader().load(str(tmp_path)))
        assert "a.hl7" in str(excinfo.value) and "message 2" in str(excinfo.value)
        assert "SECRET-CONTENT" not in str(excinfo.value)
        assert "SECRET-CONTENT" not in repr(excinfo.value.context)

    def test_an_undecodable_file(self, tmp_path: Path) -> None:
        (tmp_path / "a.hl7").write_bytes(b"\xff\xfe\x00bad")
        with pytest.raises(LoaderError, match="Failed to read"):
            list(HL7v2Loader().load(str(tmp_path)))

    def test_the_encoding_is_configurable(self, tmp_path: Path) -> None:
        text = _message("ADT^A01", _pid()).replace("HOSP", "HÖSP")
        (tmp_path / "a.hl7").write_bytes(text.encode("latin-1"))
        (record,) = _load(str(tmp_path), encoding="latin-1")
        assert record.patient_id == "PT0001"

    def test_constructing_does_no_io(self) -> None:
        HL7v2Loader(default_utc_offset_hours=-5)  # no source, no read, no import
