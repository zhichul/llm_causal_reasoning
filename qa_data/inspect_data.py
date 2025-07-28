from utils import read_jsonl

def pretty_print(obj, indent=0, indent_unit=2):
    if isinstance(obj, list):
        for item in obj:
            pretty_print(item, indent=indent, indent_unit=indent_unit)
    elif isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, str | int | float):
                print(" " * indent * indent_unit + key, end=" ")
                if isinstance(val, str):
                    print(val.replace("\n", "\n"+" " * indent * indent_unit))
                else:
                    assert isinstance(val, int) or isinstance(val, float)
                    print(str(val))
            else:
                print(" " * indent * indent_unit + key)
                pretty_print(val, indent=indent+1, indent_unit=indent_unit)
    else:
        if isinstance(obj, str):
            print(" " * indent * indent_unit + obj.replace("\n", "\n"+" " * indent * indent_unit))
        else:
            assert isinstance(obj, int) or isinstance(obj, float)
            print(" " * indent * indent_unit + str(obj))

if __name__ == "__main__":
    import argparse
    parser =  argparse.ArgumentParser()
    parser.add_argument(
        "data_file"
    )
    parser.add_argument(
        "--n", type=int, default=1
    )
    args = parser.parse_args()
    examples = read_jsonl(args.data_file, first=args.n)
    for ex in examples:
        pretty_print(ex)