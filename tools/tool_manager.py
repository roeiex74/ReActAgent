class ToolManager:
    def __init__(self, max_llm_calls: int = 10, max_tool_calls: int = 10):
        self.llm_calls = 0
        self.tool_calls = 0
        self.max_llm_calls = max_llm_calls
        self.max_tool_calls = max_tool_calls

    def can_call_llm(self):
        return self.llm_calls < self.max_llm_calls

    def can_call_tool(self):
        return self.tool_calls < self.max_tool_calls

    def register_llm_call(self):
        self.llm_calls += 1
        print(f"LLM call #{self.llm_calls}")

    def register_tool_call(self, tool_name: str):
        self.tool_calls += 1
        print(f"Tool call #{self.tool_calls}: {tool_name}")

    def summary(self):
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
        }
