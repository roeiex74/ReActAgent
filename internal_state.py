import os
import json


class InternalState:
    def __init__(self, debug: bool = False):
        self.messages = []  # Full chat history for LLM
        self.llm_calls = 0
        self.tool_calls = 0
        self.max_llm_calls = 10
        self.max_tool_calls = 10
        self.query_name = None
        self.query_text = ""
        self.log_path = None
        self.file_resources = {}
        self.done = False
        self.final_answer = None

        self.debug = debug
        self.tool_history = []  # list of (name, args)
        self.reflection_log = []  # list of reflections (strings)
        self.error_log = []  # list of error dicts
        self.state_log = {}  # milestone tracker

    def set_query_data(self, query_file_name: str):
        self.query_name = os.path.splitext(os.path.basename(query_file_name))[
            0
        ]
        self.log_path = f"{self.query_name}.log"
        with open(query_file_name, "r") as f:
            self.query_text = f.read()

    def log(self, text: str):
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(text + "\n")
        print(text)

    def can_continue(self):
        return (
            self.llm_calls < self.max_llm_calls
            and self.tool_calls < self.max_tool_calls
            and not self.done
        )

    # --- LLM and tool tracking ---
    def register_llm_call(self):
        self.llm_calls += 1

    def register_tool_call(self, tool_name: str, args: dict):
        self.tool_calls += 1
        self.tool_history.append((tool_name, args))
        self.log(f"** Entering tool {tool_name} **")

    def exit_tool(self, tool_name: str):
        self.log(f"** Exiting tool {tool_name} **")

    def already_called_tool(self, tool_name: str, args: dict):
        return any(
            tool_name == name and args == a for name, a in self.tool_history
        )

    # --- Reflection logging ---
    def add_reflection(self, text: str, log: bool = True):
        self.reflection_log.append(text)
        if log:
            self.log(f"[Reflection] {text}")
        self.messages.append(
            {"role": "assistant", "content": f"Reflection: {text}"}
        )

    # --- Error logging ---
    def log_error(self, tool_name: str, error: str, input_args: dict):
        self.error_log.append(
            {"tool": tool_name, "error": error, "args": input_args}
        )
        self.log(f"[Error] {tool_name} failed: {error}")

    # --- Final answer management ---
    def register_final_answer(self, answer_text: str):
        self.done = True
        self.final_answer = answer_text
        self.messages.append({"role": "assistant", "content": answer_text})
        self.log(f"{answer_text}")
