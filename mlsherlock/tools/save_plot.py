"""Tool: save_plot — save the current matplotlib figure to output_dir."""

import os


def run(filename, state, executor, callbacks) -> str:
    os.makedirs(state.output_dir, exist_ok=True)
    safe_name = os.path.basename(filename)
    if not safe_name.endswith((".png", ".jpg", ".pdf", ".svg")):
        safe_name += ".png"

    full_path = os.path.join(state.output_dir, safe_name)

    save_code = f"""
import os
os.makedirs({state.output_dir!r}, exist_ok=True)
plt.tight_layout()
plt.savefig({full_path!r}, dpi=120, bbox_inches='tight')
plt.clf()
print("saved")
"""
    output, error = executor.execute(save_code)
    if error:
        return f"[save_plot error] {error}"

    callbacks.on_plot(full_path)
    return f"Plot saved to: {full_path}"
