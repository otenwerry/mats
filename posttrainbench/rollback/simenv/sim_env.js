// OpenCode plugin: route bash tool calls through the local environment-
// simulator daemon (LLM-as-environment experiments).
//
// Loaded via the job home's global opencode config:
//   { "plugin": ["file:///abs/path/to/sim_env.js"] }
//
// Contract with the daemon (HTTP, default http://127.0.0.1:8378):
//   POST /route    {callID, command, description, timeout}
//     -> {mode: "real"}                       run the command unchanged
//     -> {mode: "simulate", run_instead: "true"}
//                                             neuter the real execution, then
//                                             fetch the simulated result in
//                                             the after-hook
//   POST /simulate {callID}
//     -> {output, title, metadata}            replaces the tool result
//
// The daemon decides what is "GPU-ish" (training, eval, nvidia-smi, sleep,
// timer.sh, ...), owns the virtual clock and the artifact ledger, and (in the
// real daemon) calls the simulator model. This plugin is deliberately dumb:
// no routing logic lives here.
//
// Fail-closed: if the daemon is unreachable, the tool call FAILS rather than
// silently executing a command that was meant to be simulated.

const DAEMON = process.env.SIM_DAEMON_URL || "http://127.0.0.1:8378";

async function post(path, body) {
  const res = await fetch(DAEMON + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`sim daemon ${path} -> HTTP ${res.status}`);
  return res.json();
}

export const SimEnvPlugin = async () => {
  const simulated = new Set(); // callIDs routed to the simulator

  return {
    "tool.execute.before": async (input, output) => {
      if (input.tool !== "bash") return;
      const route = await post("/route", {
        callID: input.callID,
        command: output.args?.command ?? "",
        description: output.args?.description ?? "",
        timeout: output.args?.timeout,
      });
      if (route.mode === "simulate") {
        simulated.add(input.callID);
        // run a no-op in place of the real command; the after-hook swaps in
        // the simulated result. Keeps opencode's execution machinery happy.
        output.args.command = route.run_instead ?? "true";
        if (output.args.timeout) output.args.timeout = 5000;
      }
    },

    "tool.execute.after": async (input, output) => {
      if (input.tool !== "bash" || !simulated.has(input.callID)) return;
      simulated.delete(input.callID);
      const sim = await post("/simulate", { callID: input.callID });
      output.output = sim.output;
      if (sim.title) output.title = sim.title;
      output.metadata = sim.metadata ?? {
        output: sim.output,
        exit: sim.exit ?? 0,
        description: output.metadata?.description ?? "",
        truncated: false,
        simulated: true,
      };
    },
  };
};

export default SimEnvPlugin;
