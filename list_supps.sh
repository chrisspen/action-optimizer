#!/bin/bash
.env/bin/python -m action_optimizer.optimizer list $ACTION_OPTIMIZER_DATAFILE --tags=supplement,hormone,blood --excludes=_monthly,_brand
