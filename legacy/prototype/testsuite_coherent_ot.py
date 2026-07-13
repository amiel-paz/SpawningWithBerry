#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, math
import numpy as np

from bundle import *
from models import *
from classprop import *
from propagate import *          # your code; Heff path unused here
from testutils import *

# -------------------- user knobs --------------------
h              = 5.0e-2
nstep          = 6000

# -------------------- model & initial state --------------------
model     = BerryModel2DParallelTransport(0.02, 3.0, 5.0)
init_rvec = np.asarray([-3.0, 0.0])
init_pvec = np.asarray([20.0,  0.0])

# ==============================================================
#                        Main run
# ==============================================================

if __name__ == "__main__":
    # Bundle & one initial TBF
    ndim = 2
    bundle = Bundle(ndim, model)
    init_state = 1
    bundle.add_trajectory(init_rvec, init_pvec, init_state)
    TBF = bundle.trajectorylist[0]
    rng = np.random.default_rng(1234)
