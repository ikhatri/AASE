# Copyright 2019-2020 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

# A short script to rebalance weights from 1hz to 10hz

import click


def do_thing(p1: float = 0.0, p2: float = 0.0, p3: float = 0.0):
    p1n = p1 ** (1 / 10)
    p2n = p2 / (sum([p1n ** i for i in range(10)]))
    p3n = p3 / (sum([p1n ** i for i in range(10)]))
    print(p1n, p2n, p3n, sum([p1n, p2n, p3n]))


@click.command()
@click.argument("p1")
@click.argument("p2")
@click.argument("p3")
def main(p1, p2, p3):
    do_thing(float(p1), float(p2), float(p3))


if __name__ == "__main__":
    main()
