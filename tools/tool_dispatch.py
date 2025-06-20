from tools.tool_handlers import (
    handle_online_search,
    handle_write_file,
    handle_extract_entities_from_file,
    handle_gen_plot_prog,
)

tool_dispatch = {
    "online_search": handle_online_search,
    "write_file": handle_write_file,
    "extract_entities_from_file": handle_extract_entities_from_file,
    "gen_plot_prog": handle_gen_plot_prog,
}
