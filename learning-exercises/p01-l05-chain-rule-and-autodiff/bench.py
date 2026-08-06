"""Входные данные для замера скорости."""

BENCH = {
    "chain": (2.0, 3.0),
    "chain_many": ([1.01] * 500,),
    "sigmoid": (0.7,),
    "d_sigmoid": (0.7,),
    "relu": (0.7,),
    "d_relu": (0.7,),
    "forward_backward": (2.0, 3.0, 4.0),
}
