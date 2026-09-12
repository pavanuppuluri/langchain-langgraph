import os
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

search = TavilySearch(
    max_results=3,
    include_answer=True,
    search_depth="basic",
)

agent = create_agent(
    model="openai:gpt-4o-mini",          # or "anthropic:claude-sonnet-4-6"
    tools=[search],
    system_prompt=(
        "You are a weather assistant. When asked about a city, call the search "
        "tool to find the CURRENT weather there. Report temperature (°C and °F), "
        "conditions, humidity, and wind. Cite the source URL. "
        "If the results look stale, say so rather than guessing."
    ),
)


def get_weather(city: str) -> str:
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": f"What's the weather in {city} right now?"}
        ]
    })
    return result["messages"][-1].content


if __name__ == "__main__":
    print(get_weather("Hyderabad"))
