"""SN33 miner optimization layer.

Two consumers share this package:
  * ``sn33.pipeline``  - the production miner path (latency bounded, never raises)
  * ``bench/``         - the offline harness that scores candidate strategies

Everything here is additive; the upstream repo is only touched at one hook
point (``MinerLib.do_mining``) so ``git pull`` stays clean.
"""
