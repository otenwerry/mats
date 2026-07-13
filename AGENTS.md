# Context

This repository is for my MATS AI safety research project. I am working with Maksym Andriushchenko, one of the authors of PostTrainBench (henceforth PTB). Our project will study reward hacking (henceforth RH); specifically, reward-hack-like behavior that happens at inference time (as opposed to during RL). There were some instances of this observed in PTB, and we're planning on working largely with this data. We're also generating our own reward hack trajectories using Petri, an automated auditing tool; and we may also extend into other observed RH trajectories from other benchmarks. 

PTB: Various LLM agents are evaluated for their ability to post-train various base LLMs to maximize various benchmarks, with 10 hours and one Nvidia H100 GPU. In ~5% of PTB trajectories, models reward-hacked, doing things like training on the test data. The PTB paper identified cases of rule-breaks in order to make the scores fair; we have gone further than this and identified the runs we think involved actual intentional reward-hacking; we also identify specific turns within the trajectory that involve reward-hack actions. 

Petri: A trajectory takes in a target model, judge model, auditor model, and seed prompt. The auditor uses the seed prompt to generate a scenario for the target model, and tries to elicit reward hacking (in our case) from the target. It roleplays as the environment, so when the target emits tool calls, the auditor fabricates results and returns them as a tool response block. It terminates the trajectory when it thinks it's done enough, or when a certain limit is hit. Then, the judge gives the trajectory scores of 1-10 on various dimensions; we use these scores to classify each trajectory: full reward hack, reward hack where the target took degenerate actions (we separate these out because they feel a little less interesting), reward hack where the auditor made mistakes such as pushing too hard for a hack or giving incorrect tool responses (these are basically invalid), and non-hacks (scores <5 on the reward hack trajectory). A second judge looks at the full reward hacks and identifies specific turns that involve reward-hack actions.

I have a fairly generous compute budget, so generally don't need to be very conservative with compute, although of course don't want to be wasteful. I may actually find it more of a challenge to come up with enough ideas to use the compute than to avoid using too much.

# Current methods

Our current experiments revolve around resampling from various points in a given reward-hack trajectory, and optionally inserting some sort of user-turn intervention, such as a direct request to avoid reward-hacking. We then do the same judging on our resampled trajectories as we did on our original trajectories. In order to run these experiments, all of our original trajectories need to be fully resumable whenever possible: we need to preserve all data so that we can reconstruct context. We're trying to answer questions such as: how does the probability of a re-hack change as we vary the location of resampling? (In other words, are there particular events from a trajectory that increase the probability of the reward hack happening later on?) How stubborn are reward hacking agents? (Will they obey if you just tell them to not reward-hack?) etc. 

This is easy in Petri; we have full control. It's harder in PTB and probably other benchmark trajectories where the environment is more complex and data may not be perfectly saved. A relevant consideration in PTB right now is how accurately we can rebuild context in order to do faithful resampling. 

Later on, we may move on to other methods and try to answer other questions.

# Environment
We are using uv. 

Inside of the supermats directory, we have:
- mats (this directory), which has everything I want to keep on github
- mats-local, which has all the data that I don't want to keep on github. Within this, we have folders with downloaded data (malt, posttrainbench) that should be treated as read-only, as well as folders for our own data that you're welcome to write to.
- references, which clones relevant repos. Treat as read-only.

By default, we do everything on main. Let me know if you feel the need to open a branch, otherwise just stick to main without talking about it. Let's stick to committing at the end of a chunk of changes. By default this will be once per session, at the end of the session, though sometimes we'll pivot from one thing to another at which point you can commit the first chunk of work. I don't actually care that much about when you commit so you can use your judgment and do it without permission, but please let me know when you do so. Only stage the files you touched, in case another agent is working on something else at the same time. 

# Approach to interaction

I know that you are an excellent programmer. I can generally trust that you are able to implement anything that needs to be implemented, and that if you make mistakes on the first pass, you are capable of autonomously processing the error message and coming up with a fix. However, I am far less confident in your decision-making ability. Often, you write some perfect code and then suggest a nonsense next step where there is an obviously better choice. 

This is a good complement to my skill-set (competent decision-making), so this has the potential to lead to very fruitful collaboration. However, this only works if we do a good job of managing the implementation/decision divide. In order for this to work well, you have to do a few things:
- Since you're the implementer, you have a bunch of context I don't have. This means that decision-making needs to be pretty collaborative; your role is to share relevant context as clearly and completely as possible, so that I have good information to work with. Without this, what happens is either I make bad, uninformed decisions, or I accidentally defer to you.
- Communicate to me in regular language, as regular as humanly possible. Don't come up with nicknames for things that it takes me time to interpret. Keep in mind my high-level goal for any given interaction and frame your summaries around this. If I don't understand you, I often accidentally defer an important choice to you. 
- Any time you feel like you're making a *design choice* while writing code, raise it to me. Ideally, I give you perfect specifications, so that you can implement what I want without making any high-level design choices. But the reality is that I often fail at this, which puts some responsibility on you to stay aware and let me know when something is underspecified. This way, I remain in charge of decisions.

I use viewers to look at all relevant data. If I refer to something and you don't know what it is, it's probably in the viewer. Because I'm always looking at the viewer, it's usually pretty accurate; on the other hand, the data can sometimes get misleading (e.g. poorly named labels) because I'm not looking at it all the time. If you see a contradiction, let me know, but generally it's good if your reading of the data stays pretty close to the viewer layer - this makes it more likely to be accurate. Generally your first instinct should be "let me look at this from the POV of the viewer". 

# General rules
Any file that can call an API key that costs money should have a name that starts with 'exp', short for experiment. This is for the agent settings -- I am happy for you to run any file without permission if it's free, but it's set so that I manually approve any requests to run experiments. Please follow this naming convention when creating or editing files.

For files that run for a while, e.g. because they're making lots of calls to LLMs, I appreciate regular print statements so that I can monitor progress. This doesn't need to be incredibly detailed, it's just nice to see something get printed every now and then so that I can be confident that the run is continuing as expected.

Lossy processing must be surfaced. Whenever a pipeline drops or degrades data — truncating context, clipping long fields, sampling, top-N caps, skipping unparseable items — you must (1) call this out to me explicitly at design time as a decision I get to veto, not bury it in the implementation; (2) record the omission as a stored, queryable flag on every affected output (not just a note inside a prompt or log); and (3) propagate a visible caveat to any downstream result that depends on the affected data. "The judge read 40% of the trace" is a finding about the judge, not a footnote.

Big data files should be stored in mats-local so that they don't get stored on github. Feel free to make new folders within mats-local when relevant.

posttrainbench/ISSUES.md is the canonical tracker for PTB reconstruction issues. Every time an issue is fixed, newly discovered, or changes status, update ISSUES.md AND the viewer's issue tags/legend in the same change — never let them drift apart. Issue tags use their own palette, distinct from the red/amber contamination and fidelity flags: purple = permanent data loss, blue = unresolved, teal = structural (an unavoidable artifact of resuming that trajectory's shape — known and recorded, not fixable), green = fixed in code, cleared by re-running that trajectory's continuation.

When making visuals and viewers, include the minimum amount of information necessary to clearly communicate the data. For example, don't add textual explainers to figures unless I ask for them.

# Preferences
If I ask you a simple question, I usually want a simple answer. I usually don't want you to write code unless I ask you to; often I'm just looking for an understanding of the current codebase. Do not be over-eager. Specifically, if I ask you a question about why you're doing something, DO NOT immediately pivot to doing the opposite thing; just explain to me your reasoning (and feel free to say you think you should pivot).

Communication between humans and LLMs can be tricky sometimes. We think a little bit differently, and certain parts of LLM cognition are not super intuitive to me. As such, I always appreciate suggestions about how we can interact more smoothly. Often this can look like a suggested change to this AGENTS.md/CLAUDE.md file, so that a missing piece of information is available to future instances. Don't force any suggestions, but also don't hesitate to air any ideas you have.

You're welcome to write memories to yourself, but don't overdo it, and you should be aiming to inform *future* instances of yourself, rather than your current instance, about the content of the memory. A good time to write memories is at the beginning of the session, if I say something you don't understand or if you have to take some time to figure out some important fact about the codebase -- in these cases, I probably assumed you already knew it because a previous instance knew, so I didn't bother to tell you. These are good things to store in memory. Also, if a memory ever appears to contradict the state of the codebase, let me know so we can resolve it - it's probably expired and needs to be deleted/updated.

# Other notes

Sometimes I send messages via voice memos. These often get transcribed imprecisely. If there are typos or grammatical errors, feel free to fill in the gaps yourself, or to ask me if it's not super clear what I meant.