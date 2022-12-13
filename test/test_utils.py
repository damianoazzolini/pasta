def almostEqual(a : float, b : float, digits : int) -> bool:
    epsilon = 10 ** -digits
    if b == 0:
        return a == b
    else:
        return abs(a/b - 1) < epsilon


def check_if_lists_equal(list_1, list_2):

    new_list_1 = []
    new_list_2 = []

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
        