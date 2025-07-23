from typing import Literal


# TODO: add more surveys
Dataset = Literal["acs/acs5", "dec/sf3", "geoinfo"] | str

# TODO: add more geographies
Geography = Literal[
    "us", "region", "division", "state", "county", "tract", "block group"
]

AcsVersion = Literal["acs1", "acs3", "acs5"]
