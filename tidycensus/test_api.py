from tidycensus import Census


VARS = ["B19013_001E", "B19013A_001E"]
# variety of types (trailing E, M, no trailing char)
ACS_VARS = ["B19013A_001", "B19013B_001M", "B19013C_001E"]

GEO = "state"
YEARS = [2010, 2012]

API = Census(cache_verbosity=2)


def test_get_variables():
    df = API.get_variables(
        "acs/acs5",
        YEARS,
        VARS,
        geography=GEO,
        include_metadata=False,
    )

    assert df.columns == ["year", GEO, "variable", "value"]


def test_get_variables_metatdata():
    df = API.get_variables(
        "acs/acs5",
        YEARS,
        VARS,
        geography=GEO,
        include_metadata=True,
    )

    assert df.columns == ["year", GEO, "concept", "label", "variable", "value"]


def test_acs():
    df = API.acs(
        variables=ACS_VARS,
        geography=GEO,
        include_ses=False,
        include_metadata=False,
        years=YEARS,
    )

    assert df.columns == ["year", GEO, "variable", "value"]


def test_acs_metadata_ses():
    df = API.acs(
        variables=ACS_VARS,
        geography=GEO,
        include_ses=True,
        include_metadata=True,
    )

    assert df.columns == ["year", GEO, "concept", "label", "variable", "value", "se"]
