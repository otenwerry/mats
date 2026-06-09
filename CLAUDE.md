# Context

This repository is for my MATS research project. MATS is an AI safety research fellowship that pairs me with a mentor and supports me in working on an independent research project for 3 months.

I am working with Maksym Andriushchenko, one of the authors of PostTrainBench. Our project will study the generalized reward hacking (i.e. reward-hacking behavior that happens at inference time rather than actually during RL) that was observed in the PostTrainBench data. In PostTrainBench, various LLM agents are evaluated for their ability to fine-tune various base LLMs to maximize various benchmarks, with 10 hours and one Nvidia H100 GPU. In ~5% of PTB trajectories, models reward-hacked, doing things like training on the test data. The PTB paper identified cases of rule-breaks in order to make the scores fair, but didn't go any further than that -- our goal is to advance our understanding of these naturally occurring cases of GRH.

I have a fairly generous compute budget, so generally don't need to be very conservative with compute, although of course don't want to be wasteful. I may actually find it more of a challenge to come up with enough ideas to use the compute than to avoid using too much.

# Environment
We are using uv. 

Inside of the supermats directory, we have:
- mats (this directory), which has everything I want to keep on github
- mats-local, which has all the data that I don't want to keep on github. Treat mats-local as read-only; I'm just using it for large datasets I downloaded. 
- PostTrainBench, which has the code for 

# General rules
Any file that can call an API key that costs money should have a name that starts with 'exp', short for experiment. This is for the Claude settings -- I am happy for you to run any file without permission if it's free, but it's set so that I manually approve any requests to run experiments. Please follow this naming convention when creating or editing files.

For files that run for a while, e.g. because they're making lots of calls to LLMs, I appreciate regular print statements so that I can monitor progress. This doesn't need to be incredibly detailed, it's just nice to see something get printed every now and then so that I can be confident that the run is continuing as expected.

Lossy processing must be surfaced. Whenever a pipeline drops or degrades data — truncating context, clipping long fields, sampling, top-N caps, skipping unparseable items — Claude must (1) call this out to me explicitly at design time as a decision I get to veto, not bury it in the implementation; (2) record the omission as a stored, queryable flag on every affected output (not just a note inside a prompt or log); and (3) propagate a visible caveat to any downstream result that depends on the affected data. "The judge read 40% of the trace" is a finding about the judge, not a footnote.

Generally my settings aim to allow you to take as many harmless actions as possible without permission, while requiring permission for potentially destructive actions. If you find that you had to ask for permission for something harmless where it would be easy to encode permanent permission into the settings, let me know. Similarly, if the rules restrict you from reasonable actions, let me know.

I like to deal with git myself. There's no need for you to commit anything. You're welcome to do read-only git commands to figure out answers to whatever questions you might have, but I can handle the committing.

# Preferences
Communication between humans and LLMs can be tricky sometimes. We think a little bit differently, and certain parts of LLM cognition are not super intuitive to me. As such, I always appreciate suggestions about how we can interact more smoothly. Often this can look like a suggested change to this CLAUDE.md file, so that a missing piece of information is available to future instances. Don't force any suggestions, but also don't hesitate to air any ideas you have.

You're welcome to write memories to yourself, but don't overdo it, and you should be aiming to inform *future* instances of yourself, rather than your current instance, about the content of the memory. A good time to write memories is at the beginning of the session, if I say something you don't understand or if you have to take some time to figure out some important fact about the codebase -- in these cases, I probably assumed you already knew it because a previous instance knew, so I didn't bother to tell you. These are good things to store in memory.