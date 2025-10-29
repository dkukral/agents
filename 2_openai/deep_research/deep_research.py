import gradio as gr
from dotenv import load_dotenv
from research_manager import research_manager
from agents import Runner

load_dotenv(override=True)

# TODO: Improve the status updates to be more detailed and user-friendly

async def run(query: str):
    runner = Runner.run_streamed(research_manager, query)
    status_updates = []
    
    async for event in runner.stream_events():
        # Capture tool call events
        if event.type == "tool_call_event":
            tool_name = event.data.tool_call.function.name if hasattr(event.data, 'tool_call') else "Unknown Tool"
            status_update = f"🔧 **Calling tool:** {tool_name}\n"
            status_updates.append(status_update)
            yield "\n".join(status_updates)
        
        # Capture tool result events
        elif event.type == "tool_result_event":
            tool_name = event.data.tool_call.function.name if hasattr(event.data, 'tool_call') else "Unknown Tool"
            status_update = f"✅ **Completed:** {tool_name}\n"
            status_updates.append(status_update)
            yield "\n".join(status_updates)
        
        # Capture any text output
        elif hasattr(event, 'data') and hasattr(event.data, 'delta'):
            if event.data.delta:
                status_updates.append(event.data.delta)
                yield "\n".join(status_updates)
        
        # Capture string data
        elif hasattr(event, 'data') and isinstance(event.data, str):
            status_updates.append(event.data)
            yield "\n".join(status_updates)


with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Deep Research")
    query_textbox = gr.Textbox(label="What topic would you like to research?")
    run_button = gr.Button("Run", variant="primary")
    report = gr.Markdown(label="Report")
    
    run_button.click(fn=run, inputs=query_textbox, outputs=report)
    query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)

ui.launch(inbrowser=True)

