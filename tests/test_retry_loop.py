import unittest

import agent_graph


class _FakeResp:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return _FakeResp(
            "```python\nimport cadquery as cq\nresult = cq.Workplane(\"XY\").box(1, 1, 1)\n```"
        )


class RetryLoopTests(unittest.TestCase):
    def test_coder_prompt_includes_previous_error(self):
        original_create_llm = agent_graph.create_llm
        try:
            fake = _FakeLLM()
            agent_graph.create_llm = lambda: fake

            state = {
                "input": "测试输入",
                "plan": "按计划建模",
                "errors": "RuntimeError: previous fail",
                "history": [],
                "iterations": 2,
            }
            out = agent_graph.call_coder(state)

            joined = "\n".join(getattr(m, "content", "") for m in fake.messages)
            self.assertIn("[上一轮执行错误（如有）]", joined)
            self.assertIn("RuntimeError: previous fail", joined)
            self.assertIn("重试轮次", out.get("history", [])[-1])
        finally:
            agent_graph.create_llm = original_create_llm

    def test_retry_gate_reaches_fail_after_three_retries(self):
        state = {"errors": "boom", "iterations": 0}
        decisions = []
        while True:
            decision = agent_graph.decide_to_retry(state)
            decisions.append(decision)
            if decision == "retry":
                state["iterations"] += 1
                continue
            break

        self.assertEqual(decisions, ["retry", "retry", "retry", "fail"])
        self.assertEqual(state["iterations"], 3)


if __name__ == "__main__":
    unittest.main()
