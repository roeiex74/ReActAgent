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
        self.knowledge_base = {}  # knowledge base

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
        # print(text)

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

    # --- Observation logging ---
    def add_observation(self, tool_name: str, result: str, log: bool = True):
        obs_msg = f"Observation: The tool `{tool_name}` returned:\n{result}"
        self.messages.append({"role": "assistant", "content": obs_msg})
        if log:
            self.log(obs_msg)

    # --- Reflection logging ---
    def add_reflection(self, text: str):
        self.reflection_log.append(text)
        self.log(f"[Reflection] {text}")
        self.messages.append(
            {"role": "assistant", "content": f"Reflection: {text}"}
        )

    def reflect_on_tool_error(
        self,
        tool_name: str,
        error_msg: str,
        custom_reflection: str = None,
        input_args: dict = None,
    ):
        self.log_error(tool_name, error_msg, input_args or {})
        self.add_observation(tool_name, f"Error:\n```{error_msg}```")
        reflection = (
            custom_reflection
            if custom_reflection
            else f"The tool `{tool_name}` failed. I should consider modifying the inputs, retrying, or trying a different tool."
        )
        self.add_reflection(reflection)

    # --- Final answer management ---
    def register_final_answer(self, answer_text: str):
        self.done = True
        self.final_answer = answer_text
        self.messages.append({"role": "assistant", "content": answer_text})

    def log_error(self, tool_name: str, error: str, input_args: dict):
        self.error_log.append(
            {"tool": tool_name, "error": error, "args": input_args}
        )
        self.log(f"[Error] {tool_name} failed: {error}")

    # --- ReAct next step priming ---
    def add_next_step_prompt(self):

        # Get recent tool results from reflections
        recent_tool_reflections = [
            reflection
            for reflection in self.reflection_log[-5:]  # Last 5 reflections
            if any(
                word in reflection.lower()
                for word in ["tool", "executed", "failed", "skipped"]
            )
        ]

        # # Build the next step prompt content
        # knowledge_info = ""
        # if self.knowledge_base:
        #     knowledge_info = f"\n\nKNOWLEDGE BASE (extracted data):\n{json.dumps(self.knowledge_base, indent=2)}"

        # tool_results_info = ""
        # if recent_tool_reflections:
        #     tool_results_info = f"\n\nRECENT TOOL RESULTS:\n" + "\n".join(
        #         [f"- {r}" for r in recent_tool_reflections]
        #     )

        # prompt_content = (
        #     f"{knowledge_info}{tool_results_info}\n"
        #     "Given the above observation and prior steps, think carefully about the next action.\n"
        #     "You have access to the given knowledge base and recent tool results if needed and you can extract information from them.\n"
        #     "If you have completed the task and all required information has been gathered, produce your final answer with 'Final Answer: ...'.\n"
        #     "Otherwise, explain your reasoning and choose the next appropriate tool or step.\n"
        #     "In case you encounter an error, understand the error and try to fix it using the knowledge base,given tools and reasoning logic.\n"
        #     "If you feel you cannot solve the task with the tools available to you, or that an error is preventing you from solving the task and you cannot fix it, respond with 'Final Answer: I cannot solve the task with the tools available to me.'"
        # )

        # self.messages.append(
        #     {
        #         "role": "assistant",
        #         "content": prompt_content,
        #     }
        # )
        # --- Knowledge Summary ---

        if self.knowledge_base:
            knowledge_summary = "\n".join(
                f"- {k}: {v}" for k, v in self.knowledge_base.items()
            )
        else:
            knowledge_summary = "None"

        # --- Recent Tool Results ---
        if recent_tool_reflections:
            tool_summary = "\n".join(f"- {r}" for r in recent_tool_reflections)
        else:
            tool_summary = "None"

        # --- Available File Resources ---
        if self.file_resources:
            file_summary = "\n".join(
                f"- {fn}: {desc}" for fn, desc in self.file_resources.items()
            )
        else:
            file_summary = "None"

        # --- Final Prompt Assembly ---
        prompt_content = (
            f"=== KNOWLEDGE BASE ===\n{knowledge_summary}\n\n"
            f"=== RECENT TOOL RESULTS ===\n{tool_summary}\n\n"
            f"=== AVAILABLE FILE RESOURCES ===\n{file_summary}\n\n"
            "You are solving this task step by step using the available tools.\n"
            "Leverage knowledge base and file resources to avoid redundant actions.\n"
            "Use recent tool results to guide your decisions.\n"
            "\n"
            "If a tool result indicated an error, attempt to correct the issue by:\n"
            "- Adjusting tool inputs using known facts.\n"
            "- Reusing available file resources.\n"
            "- Exploring alternative tools logically.\n"
            "- Using a debug and regenerate tool to debug the error and reproduce a corrected program in case of python program execution.\n"
            "If the task is complete, respond with 'Final Answer: ...'.\n"
            "If you are blocked and cannot proceed, respond with 'Final Answer: I cannot solve the task with the tools available to me.'\n"
            "Otherwise, explain your reasoning step by step and choose the next appropriate tool or step."
        )

        self.messages.append({"role": "assistant", "content": prompt_content})

    def update_knowledge(self, key, value):
        self.knowledge_base[key] = value
        self.add_reflection(f"Updated knowledge: {key} = {value}")

    def get_knowledge(self, key):
        return self.knowledge_base.get(key)

    def get_tool_execution_history(self):
        """
        Get a formatted history of all tool executions from reflections and knowledge base.

        Returns:
            dict: Organized tool execution history
        """
        history = {
            "recent_tool_results": {},
            "recent_tool_errors": {},
            "tool_reflections": [],
            "total_tool_calls": self.tool_calls,
        }

        # Get recent results and errors from knowledge base
        for key, value in self.knowledge_base.items():
            if key.startswith("last_result_"):
                tool_name = key.replace("last_result_", "")
                history["recent_tool_results"][tool_name] = value
            elif key.startswith("last_error_"):
                tool_name = key.replace("last_error_", "")
                history["recent_tool_errors"][tool_name] = value

        # Get tool-related reflections
        history["tool_reflections"] = [
            reflection
            for reflection in self.reflection_log
            if any(
                word in reflection.lower()
                for word in ["tool", "executed", "failed", "skipped"]
            )
        ]

        return history

    def get_latest_tool_result(self, tool_name: str):
        """
        Get the most recent result for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            dict or None: Latest result data or None if not found
        """
        result_key = f"last_result_{tool_name}"
        error_key = f"last_error_{tool_name}"

        # Check for recent result first
        result = self.get_knowledge(result_key)
        if result:
            return {"type": "success", "data": result}

        # Check for recent error
        error = self.get_knowledge(error_key)
        if error:
            return {"type": "error", "data": error}

        return None
