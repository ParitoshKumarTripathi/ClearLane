from clearlane.impact import (
    offence_severity,
    parking_offences,
    parse_violation_types,
    vehicle_factor,
)


def test_violation_parser_handles_json_list() -> None:
    parsed = parse_violation_types('["WRONG PARKING", "DOUBLE PARKING"]')
    assert parsed == ["WRONG PARKING", "DOUBLE PARKING"]


def test_non_parking_offence_is_removed() -> None:
    offences = parking_offences(["NO PARKING", "DEFECTIVE NUMBER PLATE"])
    assert offences == ["NO PARKING"]


def test_double_parking_is_more_severe_than_wrong_parking() -> None:
    assert offence_severity(["DOUBLE PARKING"]) > offence_severity(["WRONG PARKING"])


def test_heavy_vehicle_factor_exceeds_scooter() -> None:
    assert vehicle_factor("HGV") > vehicle_factor("SCOOTER")
