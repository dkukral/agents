import gradio as gr
from dotenv import load_dotenv
from research_manager import research_manager
from agents import Runner, trace, gen_trace_id

load_dotenv(override=True)
print("Loading environment variables...")

# TODO: Improve the status updates to be more detailed and user-friendly

async def run(query: str):
    print(f"Running research for query: {query}")
    yield "Please wait...."
    trace_id = gen_trace_id()
    yield f"Trace ID: <a href='https://platform.openai.com/logs/trace?trace_id={trace_id}'>{trace_id}</a>\n\n"
    with trace("Research trace", trace_id):
        runner = Runner.run_streamed(research_manager, query)
        # print(query)
        async for event in runner.stream_events():  
            if event.type == "raw_response_event":
                continue
            # When the agent updates, print that
            elif event.type == "agent_updated_stream_event":
                print(f"Agent updated: {event.new_agent.name}")
                continue
            # When items are generated, print them
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    tool_name = event.item.raw_item.name
                    print(f"Calling tool: {tool_name}")
                    yield(f"Calling tool: {tool_name}")
                    
                elif event.item.type == "tool_call_output_item":
                    continue
                elif event.item.type == "message_output_item":
                    continue
                else:
                    pass  # Ignore other event types
    
    print("Hooray! I am done.")
    yield runner.final_output



    # status_updates = []
    
    # async for event in runner.stream_events():
    #     # Debug: print event details to understand structure
    #     if event.type != 'raw_response_event':
    #         print("Received event:\n\n")
    #         print(f"Event type: {type(event)}")
    #         print(f"Event attributes: {dir(event)}")
    #         if hasattr(event, 'type'):
    #             print(f"Event.type: {event.type}")
    #         if hasattr(event, 'data'):
    #             print(f"Event.data: {type(event.data)}")
            
    #     # # Handle different event types based on actual structure
    #     # if hasattr(event, 'type'):
    #     #     if event.type == "agent_message_delta":
    #     #         if hasattr(event, 'data') and hasattr(event.data, 'delta'):
    #     #             status_updates.append(event.data.delta)
    #     #             yield "\n".join(status_updates)
        #     elif event.type == "tool_call_delta":
        #         tool_name = getattr(event.data, 'name', 'Unknown Tool')
        #         status_update = f"🔧 **Calling tool:** {tool_name}\n"
        #         status_updates.append(status_update)
        #         yield "\n".join(status_updates)
        
        # # Fallback for any text content
        # elif hasattr(event, 'data') and isinstance(event.data, str):
        #     status_updates.append(event.data)
        #     yield "\n".join(status_updates)


with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Deep Research")
    query_textbox = gr.Textbox(label="What topic would you like to research?")
    run_button = gr.Button("Run", variant="primary")
    report = gr.Markdown(label="Report")
    
    run_button.click(fn=run, inputs=query_textbox, outputs=report)
    query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)

ui.launch(inbrowser=True)

