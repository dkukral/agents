from agents import Runner, trace, gen_trace_id
from agents import Agent
    
from search_agent import search_agent_tool, search_agent
from planner_agent import planner_agent_tool, planner_agent, WebSearchItem, WebSearchPlan
from writer_agent import writer_agent_tool, writer_agent, ReportData
from email_agent import email_agent_tool, email_agent
import asyncio

INSTRUCTIONS = (
    "You are a senior researcher tasked with writing a cohesive report for a research query.\n"
    "You will be provided with the original query and should use the following tools to help you gather information and write the report.\n"
    " 1. Use the 'planner_agent' tool to plan out the searches you need to perform.\n"
    " 2. Use the 'search_agent' tool to perform web searches based on the plan.\n"
    " 3. Use the 'writer_agent' tool to write the final report based on the search results.\n"
    " 4. Use the 'email_agent' tool to send the final report via email.\n"
    "The final output should be in markdown format, and it should be lengthy and detailed. Aim "
    "for 5-10 pages of content, at least 1000 words."
)

research_manager = Agent(
    name="ResearchManager",
    instructions=INSTRUCTIONS,
    tools=[
        search_agent_tool,
        planner_agent_tool,
        writer_agent_tool,
        email_agent_tool
    ],
    model="gpt-4o-mini",
)

# class ResearchManager:

#     async def run(self, query: str):
#         """ Run the deep research process, yielding the status updates and the final report"""
#         trace_id = gen_trace_id()
#         with trace("Research trace", trace_id=trace_id):
#             print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
#             yield f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
#             print("Starting research...")
#             search_plan = await self.plan_searches(query)
#             yield "Searches planned, starting to search..."     
#             search_results = await self.perform_searches(search_plan)
#             yield "Searches complete, writing report..."
#             report = await self.write_report(query, search_results)
#             yield "Report written, sending email..."
#             await self.send_email(report)
#             yield "Email sent, research complete"
#             yield report.markdown_report
        

#     async def plan_searches(self, query: str) -> WebSearchPlan:
#         """ Plan the searches to perform for the query """
#         print("Planning searches...")
#         result = await Runner.run(
#             planner_agent,
#             f"Query: {query}",
#         )
#         print(f"Will perform {len(result.final_output.searches)} searches")
#         return result.final_output_as(WebSearchPlan)

#     async def perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
#         """ Perform the searches to perform for the query """
#         print("Searching...")
#         num_completed = 0
#         tasks = [asyncio.create_task(self.search(item)) for item in search_plan.searches]
#         results = []
#         for task in asyncio.as_completed(tasks):
#             result = await task
#             if result is not None:
#                 results.append(result)
#             num_completed += 1
#             print(f"Searching... {num_completed}/{len(tasks)} completed")
#         print("Finished searching")
#         return results

#     async def search(self, item: WebSearchItem) -> str | None:
#         """ Perform a search for the query """
#         input = f"Search term: {item.query}\nReason for searching: {item.reason}"
#         try:
#             result = await Runner.run(
#                 search_agent,
#                 input,
#             )
#             return str(result.final_output)
#         except Exception:
#             return None

#     async def write_report(self, query: str, search_results: list[str]) -> ReportData:
#         """ Write the report for the query """
#         print("Thinking about report...")
#         input = f"Original query: {query}\nSummarized search results: {search_results}"
#         result = await Runner.run(
#             writer_agent,
#             input,
#         )

#         print("Finished writing report")
#         return result.final_output_as(ReportData)
    
#     async def send_email(self, report: ReportData) -> None:
#         print("Writing email...")
#         result = await Runner.run(
#             email_agent,
#             report.markdown_report,
#         )
#         print("Email sent")
#         return report