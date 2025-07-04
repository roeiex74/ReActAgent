import json
from tools.online_search_tool import internet_search_attribute
from tools.write_file_tool import write_file
from tools.extract_entities_from_file_tool import extract_entities_from_file
from tools.gen_plot_prog_tool import gen_plot_prog
from tools.debug_and_regenerate_prog import debug_and_regenerate_prog
from tools.execute_Python_prog import execute_Python_prog


def handle_online_search(**kwargs):
    an_entity = kwargs.get("an_entity")
    an_attribute = kwargs.get("an_attribute")
    max_retries = kwargs.get("max_retries", 3)

    if (
        not an_entity
        or not isinstance(an_entity, str)
        or not an_entity.strip()
    ):
        return _error(
            "Invalid or missing 'an_entity' parameter", tool="online_search"
        )

    if (
        not an_attribute
        or not isinstance(an_attribute, str)
        or not an_attribute.strip()
    ):
        return _error(
            "Invalid or missing 'an_attribute' parameter", tool="online_search"
        )

    try:
        result = internet_search_attribute(
            an_entity=an_entity,
            an_attribute=an_attribute,
            max_results=max_retries,
        )

        if not isinstance(result, str):
            return _error(
                "Tool returned invalid response type",
                tool="online_search",
                expected="string",
                received=type(result).__name__,
            )
        return result

    except Exception as e:
        return _error(
            "Tool execution failed", tool="online_search", details=str(e)
        )


def handle_write_file(**kwargs):
    file_content = kwargs.get("file_content")
    fn = kwargs.get("fn")

    if not fn or not isinstance(fn, str):
        return _error("Missing or invalid filename", tool="write_file")

    if not isinstance(file_content, str):
        return _error("file_content must be a string", tool="write_file")

    return write_file(file_content, fn)


def handle_extract_entities_from_file(**kwargs):
    file_name = kwargs.get("file_name")
    entity_type = kwargs.get("entity_type")

    if not file_name or not entity_type:
        return _error(
            "Missing required parameter", tool="extract_entities_from_file"
        )

    return extract_entities_from_file(file_name, entity_type)


def handle_gen_plot_prog(**kwargs):
    # required simple strings
    required_str = [
        "plot_request",
        "input_file",
        "columns",
        "gen_output_program_fn",
        "output_png",
    ]
    for key in required_str:
        if not kwargs.get(key):
            return _error(
                f"Missing required parameter '{key}'", tool="gen_plot_prog"
            )

    # knowledge_base must be supplied, but {} is allowed
    if "knowledge_base" not in kwargs or kwargs["knowledge_base"] is None:
        return _error(
            "Missing required parameter 'knowledge_base'", tool="gen_plot_prog"
        )

    return gen_plot_prog(
        kwargs["plot_request"],
        kwargs["input_file"],
        kwargs["columns"],
        kwargs["gen_output_program_fn"],
        kwargs["output_png"],
        kwargs["knowledge_base"],
    )


def handle_execute_Python_prog(**kwargs):
    program_fn = kwargs.get("program_fn")

    if not program_fn:
        return _error("Missing required parameter", tool="execute_Python_prog")

    return execute_Python_prog(program_fn)


def handle_debug_and_regenerate_prog(**kwargs):
    program_fn = kwargs.get("program_fn")
    errors = kwargs.get("errors")
    plot_request = kwargs.get("plot_request")
    data_file = kwargs.get("data_file")
    columns = kwargs.get("columns")

    if (
        not program_fn
        or not errors
        or not plot_request
        or not data_file
        or not columns
    ):
        return _error(
            "Missing required parameter", tool="debug_and_regenerate_prog"
        )

    return debug_and_regenerate_prog(
        program_fn, errors, plot_request, data_file, columns
    )


def handle_execute_Python_prog(**kwargs):
    program_fn = kwargs.get("program_fn")

    if not program_fn:
        return _error("Missing required parameter", tool="execute_Python_prog")

    return execute_Python_prog(program_fn)


def _error(msg, tool, **extra):
    payload = {"error": msg, "tool": tool}
    payload.update(extra)
    return json.dumps(payload)
