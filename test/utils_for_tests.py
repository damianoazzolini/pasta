def almostEqual(target : float, computed : float, tolerance : float = 0.015) -> bool:
    return (abs(target - computed) <= tolerance) and (computed >= 0) and (computed <= 1)


def check_if_lists_equal(list_1 : 'list[str]', list_2 : 'list[str]') -> bool:

    new_list_1 : 'list[str]' = []
    new_list_2 : 'list[str]' = []

    if len(list_1) != len(list_2):
        # print("Error, Lists are not the same length!")
        return False
    
    else:
        for i in range(0, len(list_1)):       

            if len(list_1[i]) != len(list_2[i]):
                # print("Error, Sublists are not the same length!")
                return False

            else:
                sorted_sublist_1 = sorted(list_1[i])
                sorted_sublist_2 = sorted(list_2[i])

                new_list_1.append(sorted_sublist_1)
                new_list_2.append(sorted_sublist_2)


    new_list_sorted_1 = sorted(new_list_1)
    new_list_sorted_2 = sorted(new_list_2)


    if new_list_sorted_1 == new_list_sorted_2:
        return True
    else:
        # print("Error, Lists are not equal")
        return False

class TestArguments:
    def __init__(self,
        test_name : str,
        filename : str,
        query : str,
        expected_lp : float,
        expected_up : float,
        evidence : str = "",
        samples : int = 5000,
        mh : bool = False,
        gibbs : bool = False,
        rejection : bool = False,
        normalize : bool = False) -> None:
        self.test_name = test_name
        self.filename = filename
        self.query = query
        self.expected_lp = expected_lp
        self.expected_up = expected_up
        self.evidence = evidence
        self.samples = samples
        self.mh = mh
        self.gibbs = gibbs
        self.rejection = rejection
        self.normalize = normalize
        
# print(almostEqual(0.5,0.484,0.015))