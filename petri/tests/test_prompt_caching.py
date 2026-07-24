from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.model._openai import openai_completion_params
from inspect_ai.tool import ToolCall

PETRI = Path(__file__).resolve().parents[1]
LIB = PETRI / "lib"
for path in (str(PETRI), str(LIB)):
    if path not in sys.path:
        sys.path.insert(0, path)

import prompt_caching


class PromptCacheBarrierTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_call_warms_then_waiters_fan_out_with_fresh_results(self):
        barrier = prompt_caching.PromptCacheWarmupBarrier()
        release_warmup = asyncio.Event()
        release_peers = asyncio.Event()
        started: list[int] = []

        async def call(i: int) -> int:
            started.append(i)
            if i == 0:
                await release_warmup.wait()
            else:
                await release_peers.wait()
            return i

        tasks = [asyncio.create_task(barrier.run("same-prefix", lambda i=i: call(i)))
                 for i in range(4)]
        await asyncio.sleep(0)
        self.assertEqual(started, [0])

        release_warmup.set()
        for _ in range(20):
            await asyncio.sleep(0)
            if len(started) == 4:
                break
        self.assertCountEqual(started, [0, 1, 2, 3])

        release_peers.set()
        self.assertEqual(await asyncio.gather(*tasks), [0, 1, 2, 3])
        self.assertEqual(barrier.warmed_prefixes, 1)
        self.assertEqual(barrier.waited_requests, 3)

    async def test_failed_warmup_allows_next_request_to_try(self):
        barrier = prompt_caching.PromptCacheWarmupBarrier()

        async def fail():
            raise RuntimeError("provider failure")

        with self.assertRaisesRegex(RuntimeError, "provider failure"):
            await barrier.run("prefix", fail)
        self.assertEqual(await barrier.run("prefix", lambda: asyncio.sleep(0, result=7)), 7)
        self.assertEqual(barrier.failed_warmups, 1)
        self.assertEqual(barrier.warmed_prefixes, 1)


class PromptCachePayloadTests(unittest.TestCase):
    def test_direct_anthropic_helpers_mark_only_reusable_blocks(self):
        system = prompt_caching.cached_system("stable system")
        user = prompt_caching.cached_user_prefix("stable prefix", "varying suffix")
        self.assertEqual(system[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(user[0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", user[1])
        self.assertEqual("".join(block["text"] for block in user),
                         "stable prefixvarying suffix")

    def test_stable_key_does_not_retain_or_confuse_prefixes(self):
        first = prompt_caching.stable_key("x", "private long prompt")
        self.assertEqual(first, prompt_caching.stable_key("x", "private long prompt"))
        self.assertNotEqual(first, prompt_caching.stable_key("x", "private long prompt!"))
        self.assertNotIn("private", first)

    def test_empty_report_is_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as td:
            report = prompt_caching.build_report(Path(td))
        self.assertEqual(report["schema_version"], 3)
        self.assertEqual(report["n_eval_logs"], 0)
        self.assertEqual(report["models"], {})
        self.assertEqual(report["verified_models"], [])
        self.assertFalse(report["inspect_response_cache_enabled"])
        self.assertEqual(
            report["warmup_barrier"]["key_version"],
            "v2-ignore-provider-invisible-chat-message-id",
        )
        self.assertEqual(
            report["warmup_barrier"]["openrouter_session_routing"],
            "v1-stable-from-conversation-opening",
        )


class InspectWarmupKeyTests(unittest.TestCase):
    model = SimpleNamespace(model_name="anthropic/test-model", base_url=None)

    def _key(self, messages):
        return prompt_caching._inspect_request_key(
            self.model,
            (messages, [], "auto", None),
            {},
        )

    def test_provider_identical_messages_ignore_inspect_message_ids(self):
        first = [
            ChatMessageSystem(id="inspect-id-a", content="same long prefix " * 300),
            ChatMessageUser(id="suffix-a", content="new task"),
        ]
        second = [
            ChatMessageSystem(id="inspect-id-b", content="same long prefix " * 300),
            ChatMessageUser(id="suffix-b", content="new task"),
        ]

        self.assertIsNotNone(self._key(first))
        self.assertEqual(self._key(first), self._key(second))

    def test_meaningful_prompt_content_still_changes_the_key(self):
        first = [
            ChatMessageSystem(id="same-id", content="long prefix A " * 300),
            ChatMessageUser(content="new task"),
        ]
        second = [
            ChatMessageSystem(id="same-id", content="long prefix B " * 300),
            ChatMessageUser(content="new task"),
        ]

        self.assertNotEqual(self._key(first), self._key(second))

    def test_provider_visible_tool_call_ids_still_change_the_key(self):
        def messages(tool_call_id: str):
            return [
                ChatMessageSystem(content="same long prefix " * 300),
                ChatMessageAssistant(
                    content="",
                    tool_calls=[ToolCall(
                        id=tool_call_id,
                        function="read_file",
                        arguments={"path": "results.csv"},
                    )],
                ),
                ChatMessageUser(content="continue"),
            ]

        self.assertNotEqual(self._key(messages("call-a")), self._key(messages("call-b")))


class OpenRouterSessionTests(unittest.TestCase):
    model = SimpleNamespace(model_name="openrouter/moonshotai/kimi", base_url=None)

    def test_session_stays_stable_across_message_ids_and_later_turns(self):
        first = [
            ChatMessageSystem(id="system-a", content="same system"),
            ChatMessageUser(id="user-a", content="same opening task"),
            ChatMessageAssistant(content="first trajectory response"),
        ]
        later = [
            ChatMessageSystem(id="system-b", content="same system"),
            ChatMessageUser(id="user-b", content="same opening task"),
            ChatMessageAssistant(content="different later response"),
            ChatMessageUser(content="different later result"),
        ]

        self.assertEqual(
            prompt_caching._openrouter_session_id(self.model, first),
            prompt_caching._openrouter_session_id(self.model, later),
        )

    def test_different_conversation_openings_get_different_sessions(self):
        first = [
            ChatMessageSystem(content="same system"),
            ChatMessageUser(content="task A"),
        ]
        second = [
            ChatMessageSystem(content="same system"),
            ChatMessageUser(content="task B"),
        ]

        self.assertNotEqual(
            prompt_caching._openrouter_session_id(self.model, first),
            prompt_caching._openrouter_session_id(self.model, second),
        )

    def test_session_is_merged_without_mutating_other_config(self):
        messages = [
            ChatMessageSystem(content="same long system " * 300),
            ChatMessageUser(content="opening task"),
        ]
        original_config = prompt_caching.GenerateConfig(
            extra_body={"provider": {"allow_fallbacks": True}},
        )
        args, kwargs = prompt_caching._with_openrouter_session(
            self.model,
            (messages, [], "auto", original_config),
            {},
        )

        self.assertEqual(kwargs, {})
        routed_config = args[3]
        self.assertNotIn("session_id", original_config.extra_body)
        self.assertEqual(
            routed_config.extra_body["provider"],
            {"allow_fallbacks": True},
        )
        self.assertRegex(
            routed_config.extra_body["session_id"],
            r"^petri-prefix-[0-9a-f]{48}$",
        )
        wire_params = openai_completion_params(
            "moonshotai/kimi",
            routed_config,
            tools=False,
        )
        self.assertEqual(
            wire_params["extra_body"]["session_id"],
            routed_config.extra_body["session_id"],
        )

    def test_explicit_session_id_is_respected(self):
        messages = [
            ChatMessageSystem(content="same long system " * 300),
            ChatMessageUser(content="opening task"),
        ]
        original_config = prompt_caching.GenerateConfig(
            extra_body={"session_id": "deliberate-session"},
        )
        args, kwargs = prompt_caching._with_openrouter_session(
            self.model,
            (messages, [], "auto", original_config),
            {},
        )

        self.assertEqual(kwargs, {})
        self.assertIs(args[3], original_config)
        self.assertEqual(args[3].extra_body["session_id"], "deliberate-session")


if __name__ == "__main__":
    unittest.main()
