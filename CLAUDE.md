This repository is the beginning of my MATS research project. Right now I'm just exploring, so there's not a lot of information to share, but here's what I have so far.

# Context

MATS is an AI safety research fellowship. I'm paired with a mentor and assigned to work on a project for a few months. Right now it seems like we'll be researching generalized reward hacking, i.e. reward-hacking behavior that happens at inference time rather than actually during RL. We're most interested in naturally emergent (not directly elicited) reward hacking; with current frontier models, it usually takes a lot of context for the model to want to do this, so we're largely looking at long-context/multi-turn/memory-based/agentic settings.

I have a compute budget of $2,000 per week, so generally don't need to be very conservative with compute, although of course don't want to be wasteful. I may actually find it more of a challenge to come up with enough ideas to use the compute than to avoid using too much.

# Environment
We are using uv.

# General rules
Any file that can call an API key that costs money should have a name that starts with 'exp', short for experiment. This is for the Claude settings -- I am happy for you to run any file without permission if it's free, but it's set so that I manually approve any requests to run experiments. Please follow this naming convention when creating or editing files.

Generally my settings aim to allow you to take as many harmless actions as possible without permission, while requiring permission for potentially destructive actions. If you find that you had to ask for permission for something harmless where it would be easy to encode permanent permission into the settings, let me know. Similarly, if the rules restrict you from reasonable actions, let me know.

I like to deal with git myself. There's no need for you to commit anything. You're welcome to do read-only git commands to figure out answers to whatever questions you might have, but I can handle the committing.

# Preferences
Communication between humans and LLMs can be tricky sometimes. We think a little bit differently, and certain parts of LLM cognition are not super intuitive to me. As such, I always appreciate suggestions about how we can interact more smoothly. Often this can look like a suggested change to this CLAUDE.md file, so that a missing piece of information is available to future instances. Don't force any suggestions, but also don't hesitate to air any ideas you have.