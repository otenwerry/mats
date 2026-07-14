# How PostTrainBench works

Various LLM agents are evaluated for their ability to post-train various base LLMs to maximize various benchmarks, with 10 hours and one Nvidia H100 GPU. In ~5% of PTB trajectories, models reward-hacked, doing things like training on the test data. The PTB paper identified cases of rule-breaks in order to make the scores fair; we have gone further than this and identified the runs we think involved actual intentional reward-hacking; we also identify specific turns within the trajectory that involve reward-hack actions. 
