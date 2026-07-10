"""Two independent reconstructions of the cut-point workspace.

forward.py  — replay the agent's tool calls from t=0 up to the cut (execute).
backward.py — take the archived FINAL workspace and roll it back to the cut.

Both start from scaffold.assemble() and should converge on the same tree; the
fidelity inspector cross-checks them.
"""
