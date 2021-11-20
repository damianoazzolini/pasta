import pasp_parser

if __name__ == "__main__":
    filename = "../test/expansion.lp"
    query = "fly(1)."
    pasp_parser = pasp_parser.PaspParser(filename)
    pasp_parser.parse()
    print(pasp_parser)