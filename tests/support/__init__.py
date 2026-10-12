"""Code the tests need that the library does not ship.

``bases`` and ``intertwiners`` are dense, real-arithmetic reference
implementations of the SO(2)-steerable kernel bases and the SO(2)
intertwiners, written for transparency rather than speed; the fast
complex-arithmetic paths in ``gm_attention.nn.kernels`` and
``gm_attention.nn.linear`` are checked against them.
``synthetic_2d3ds`` writes a random dataset in the released converter's
layout so the segmentation pipeline can be exercised without the download.
Nothing under ``src/`` imports from here.
"""
