"""Top-level entry for Gradio app used by Hugging Face Spaces and GitHub deployments.

This file exposes a `demo` variable (a Gradio app) and launches it when run locally.
It simply imports the dashboard Gradio `demo` defined in `dashboard/app.py`.
"""
from dashboard import app as dashboard_app

# Expose the Gradio Blocks instance for hosting platforms (Hugging Face Spaces expects
# a variable named `demo` or `app` pointing to the Gradio interface).
demo = getattr(dashboard_app, "demo", None)


if __name__ == "__main__":
    if demo is None:
        raise RuntimeError("demo Gradio object not found in dashboard.app")
    demo.launch(server_name="127.0.0.1", share=False)
