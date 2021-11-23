import time

# local
import pasp_parser
import asp_interface

if __name__ == "__main__":
    # filename = "../test/expansion.lp"
    # query = "a"
    filename = "../examples/bird_10.lp"
    query = "nofly(1)."
    # filename = "../examples/bird_4.lp"
    # query = "fly(1)."
    # filename = "../examples/bird_2_2.lp"
    # query = "fly_1"

    start_time = time.time()

    precision = 3
    verbose = True
    pedantic = False

    parser = pasp_parser.PaspParser(filename,10**3)
    parser.add_query(query)
    parser.parse()

    if verbose:
        print("Parsed program")

    content_find_minimal_set = parser.get_content_to_compute_minimal_prob_facts()
    asp_program = parser.get_asp_program()

    if pedantic:
        print("--- Asp program (without minimal set) ---")
        for e in asp_program:
            print(e)
        print("---")

    interface = asp_interface.AspInterface(content_find_minimal_set,asp_program)
    exec_time = interface.get_minimal_set_probabilistic_facts()

    if verbose:
        print("Computed cautious consequences in %s seconds" % (exec_time))
        if pedantic:
            print("--- Minimal set of probabilistic facts ---")
            print(interface.get_cautious_consequences())
            print("---")

    computed_models, grounding_time, computation_time, world_analysis_time = interface.compute_probabilities()
    end_time = time.time() - start_time

    if verbose:
        print("Computed models: " + str(computed_models))
        print("Grounding time (s): " + str(grounding_time))
        print("Probability computation time (s): " + str(computation_time))
        print("World analysis time (s): " + str(world_analysis_time))
        print("Total time (s): " + str(end_time))

    print("Lower probability for the query " + query + ": " + interface.get_lower_probability())
    print("Upper probability for the query " + query + ": " + interface.get_upper_probability())
