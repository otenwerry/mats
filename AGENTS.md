# Context

This repository is for my MATS research project. MATS is an AI safety research fellowship that pairs me with a mentor and supports me in working on an independent research project for 3 months.

I am working with Maksym Andriushchenko, one of the authors of PostTrainBench. Our project will study the generalized reward hacking (i.e. reward-hacking behavior that happens at inference time rather than actually during RL) that was observed in the PostTrainBench data. In PostTrainBench, various LLM agents are evaluated for their ability to fine-tune various base LLMs to maximize various benchmarks, with 10 hours and one Nvidia H100 GPU. In ~5% of PTB trajectories, models reward-hacked, doing things like training on the test data. The PTB paper identified cases of rule-breaks in order to make the scores fair, but didn't go any further than that -- our goal is to advance our understanding of these naturally occurring cases of GRH.

I have a fairly generous compute budget, so generally don't need to be very conservative with compute, although of course don't want to be wasteful. I may actually find it more of a challenge to come up with enough ideas to use the compute than to avoid using too much.

# Environment
We are using uv. 

Inside of the supermats directory, we have:
- mats (this directory), which has everything I want to keep on github
- mats-local, which has all the data that I don't want to keep on github. Within this, we have folders with downloaded data (malt, posttrainbench) that should be treated as read-only, as well as folders for our own data that you're welcome to write to.
- Other directories that clone useful repos for reference. Treat as read-only.

# Approach to interaction

I know that you are an excellent programmer. I can generally trust that you are able to implement anything that needs to be implemented, and that if you make mistakes on the first pass, you are capable of autonomously processing the error message and coming up with a fix. However, I am less confident in your decision-making ability. Often, you write some perfect code and then suggest an absolutely inane next step where there is an obvious other option that presents a clear Pareto improvement. 

This is a good complement to my skill-set (competent decision-making), so this has the potential to lead to very fruitful collaboration. However, this only works if we do a good job of managing the implementation/decision divide. In order for this to work well, you have to do a few things:
- Since you're the implementer, you have a bunch of context I don't have. This means that decision-making needs to be pretty collaborative; your role is to share relevant context as clearly and completely as possible, so that I have good information to work with. Without this, what happens is either I make bad, uninformed decisions, or I accidentally defer to you.
- Communicate to me in regular language, as regular as humanly possible. Don't come up with nicknames for things that it takes me time to interpret. Keep in mind my high-level goal for any given interaction and frame your summaries around this. If I don't understand you, I often accidentally defer an important choice to you. 
- Any time you feel like you're making a *design choice* while writing code, raise it to me. Ideally, I give you perfect specifications, so that you can implement what I want without making any high-level design choices. But the reality is that I often fail at this, which puts some responsibility on you to stay aware and let me know when something is underspecified. This way, I remain in charge of decisions.

# General rules
Any file that can call an API key that costs money should have a name that starts with 'exp', short for experiment. This is for the agent settings -- I am happy for you to run any file without permission if it's free, but it's set so that I manually approve any requests to run experiments. Please follow this naming convention when creating or editing files.

For files that run for a while, e.g. because they're making lots of calls to LLMs, I appreciate regular print statements so that I can monitor progress. This doesn't need to be incredibly detailed, it's just nice to see something get printed every now and then so that I can be confident that the run is continuing as expected.

Lossy processing must be surfaced. Whenever a pipeline drops or degrades data — truncating context, clipping long fields, sampling, top-N caps, skipping unparseable items — you must (1) call this out to me explicitly at design time as a decision I get to veto, not bury it in the implementation; (2) record the omission as a stored, queryable flag on every affected output (not just a note inside a prompt or log); and (3) propagate a visible caveat to any downstream result that depends on the affected data. "The judge read 40% of the trace" is a finding about the judge, not a footnote.

Big data files should be stored in mats-local so that they don't get stored on github. Feel free to make new folders within mats-local when relevant.

Generally my settings aim to allow you to take as many harmless actions as possible without permission, while requiring permission for potentially destructive actions. If you find that you had to ask for permission for something harmless where it would be easy to encode permanent permission into the settings, let me know. Similarly, if the rules restrict you from reasonable actions, let me know.

I like to deal with git myself. There's no need for you to commit anything. You're welcome to do read-only git commands to figure out answers to whatever questions you might have, but I can handle the committing.

When making visuals and viewers, include the minimum amount of information necessary to clearly communicate the data. For example, don't add textual explainers to figures unless I ask for them.

# Preferences
If I ask you a simple question, I usually want a simple answer. I usually don't want you to write code unless I ask you to; often I'm just looking for an understanding of the current codebase. Do not be over-eager. Specifically, if I ask you a question about why you're doing something, DO NOT immediately pivot to doing the opposite thing; just explain to me your reasoning (and feel free to say you think you should pivot).

Communication between humans and LLMs can be tricky sometimes. We think a little bit differently, and certain parts of LLM cognition are not super intuitive to me. As such, I always appreciate suggestions about how we can interact more smoothly. Often this can look like a suggested change to this AGENTS.md/CLAUDE.md file, so that a missing piece of information is available to future instances. Don't force any suggestions, but also don't hesitate to air any ideas you have.

You're welcome to write memories to yourself, but don't overdo it, and you should be aiming to inform *future* instances of yourself, rather than your current instance, about the content of the memory. A good time to write memories is at the beginning of the session, if I say something you don't understand or if you have to take some time to figure out some important fact about the codebase -- in these cases, I probably assumed you already knew it because a previous instance knew, so I didn't bother to tell you. These are good things to store in memory.