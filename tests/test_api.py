from tidycensus import Census


VARS = ["B19013_001E", "B19013A_001E"]
GROUPS = ["group(P148A)"]
# variety of types (trailing E, M, no trailing char)
ACS_VARS = ["B19013A_001", "B19013B_001M", "B19013C_001E"]

acs_median_incomes = {
    "Pooled": "B19013_001",
    "White": "B19013A_001",
    "Black": "B19013B_001",
    "AIAN": "B19013C_001",
    "Asian": "B19013D_001",
    "Non-Hispanic White": "B19013H_001",
    "Hispanic": "B19013I_001",
}

GEO = "state"
YEARS = [2010, 2012]

API = Census(cache_verbosity=2)


def test_metadata():
    df = API.get_metadata("acs/acs5", [2010])


def test_get_variables():
    df = API.get_variables(
        "acs/acs5",
        YEARS,
        VARS,
        geography=GEO,
        include_metadata=False,
    )

    assert df.columns == ["year", GEO, "variable", "value"]


def test_get_variables_group():
    df = API.get_variables(
        "dec/sf3",
        years=[2000],
        variables=["group(P148A)"],
        geography="county",
    )


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
        years=YEARS,
    )

    assert df.columns == ["year", GEO, "concept", "label", "variable", "value", "se"]

    assert df.get_column("concept").unique().len() == len(YEARS) * len(ACS_VARS)


def test_acs_county():
    df = API.acs(
        list(acs_median_incomes.values()),
        geography="county",
        years=[2010],
        include_metadata=False,
    )

    assert df.columns == ["year", "county", "variable", "value", "se"]
