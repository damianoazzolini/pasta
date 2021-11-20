import importlib.util
from typing import Union

spec = importlib.util.spec_from_file_location("pasp_parser", "../src/pasp_parser.py")
pasp_parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pasp_parser)

# TODO
# def test_parse() -> Union[int,int]:
