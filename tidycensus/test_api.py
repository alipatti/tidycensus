from tidycensus import Census

def test_acs():
    api = Census(cache_verbosity=0)

    variables = ["B19013_001E", "B19013A_001E"]
    geography = "state"
    dataset = "acs/acs5"
    years = [2010, 2012]

    df = api._get_variables(dataset, years, variables, geography=geography)
