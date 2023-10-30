from llmtuner import run_exp
# import sys

def main():
    # print(sys.argv)
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
