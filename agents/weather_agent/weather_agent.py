#!/usr/bin/env python3
"""Weather Agent using Pydantic AI and MCP servers."""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional

import click
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel

# Import logging configuration from the shared library
from agentic.logging import colorized_config

# Configure logging
colorized_config(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Configuration Models
# --------------------------------------------------------------------------------------


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server endpoint."""

    name: str = Field(..., description="Name of the MCP server")
    command: List[str] = Field(..., description="Command to run the MCP server")
    env: Optional[dict] = Field(
        default=None, description="Environment variables for the server"
    )


class WeatherAgentConfig(BaseModel):
    """Configuration for the Weather Agent."""

    mcp_servers: List[MCPServerConfig] = Field(
        default_factory=list, description="List of MCP server configurations"
    )
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_api_base: Optional[str] = Field(
        default=None, description="OpenAI API base URL"
    )
    model_name: str = Field(
        default="openai/gpt-oss-20b", description="Model name to use"
    )


# --------------------------------------------------------------------------------------
# Weather Agent Dependencies
# --------------------------------------------------------------------------------------


class WeatherAgentDeps(BaseModel):
    """Dependencies for the Weather Agent."""

    mcp_clients: dict = Field(
        default_factory=dict, description="MCP client connections"
    )


# --------------------------------------------------------------------------------------
# Weather Agent
# --------------------------------------------------------------------------------------


class WeatherAgent:
    """Weather Agent that uses Pydantic AI and MCP servers for weather information."""

    def __init__(self, config: WeatherAgentConfig):
        """Initialize the Weather Agent.

        Args:
            config: Configuration for the weather agent
        """
        self.config = config
        self.deps = WeatherAgentDeps()

        # Set up OpenAI API configuration from environment variables
        openai_api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        openai_api_base = config.openai_api_base or os.getenv(
            "OPENAI_API_BASE_URL", os.getenv("OPENAI_BASE_URL")
        )

        if not openai_api_key:
            raise ValueError(
                "OpenAI API key must be provided via config or OPENAI_API_KEY environment variable"
            )

        # Initialize the OpenAI model
        # Use environment variables for configuration
        os.environ["OPENAI_API_KEY"] = openai_api_key
        if openai_api_base:
            os.environ["OPENAI_BASE_URL"] = openai_api_base

        self.model = OpenAIChatModel(config.model_name)

        # Read system prompt from file
        system_prompt = self._load_system_prompt()

        # Initialize the Pydantic AI agent
        self.agent = Agent(
            model=self.model,
            deps_type=WeatherAgentDeps,
            system_prompt=system_prompt,
        )

        # Initialize MCP connections
        self._initialize_mcp_connections()

        # Register tools from MCP servers
        self._register_mcp_tools()

    def _load_system_prompt(self) -> str:
        """Load the system prompt from the SYSTEM_PROMPT.md file."""
        # Get the directory containing this script
        agent_dir = Path(__file__).parent
        system_prompt_path = agent_dir / "SYSTEM_PROMPT.md"

        if not system_prompt_path.exists():
            raise FileNotFoundError(
                f"System prompt file not found: {system_prompt_path}"
            )

        with open(system_prompt_path, "r", encoding="utf-8") as file:
            return file.read()

    def _initialize_mcp_connections(self) -> None:
        """Initialize connections to MCP servers."""
        for server_config in self.config.mcp_servers:
            try:
                logger.info(f"Connecting to MCP server: {server_config.name}")

                # Create stdio client for the MCP server
                # Note: MCP client creation requires different approach
                self.deps.mcp_clients[server_config.name] = {
                    "command": server_config.command,
                    "env": server_config.env or {},
                }

                logger.info(f"Successfully configured MCP server: {server_config.name}")

            except Exception as e:
                logger.error(
                    f"Failed to configure MCP server {server_config.name}: {e}"
                )
                raise

    def _register_mcp_tools(self) -> None:
        """Register tools from MCP servers with the Pydantic AI agent."""
        # For now, we'll register some basic tools manually
        # In a full implementation, we would dynamically discover tools from MCP servers

        @self.agent.tool
        async def get_weather_forecast(
            ctx: RunContext[WeatherAgentDeps],
            latitude: float,
            longitude: float,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
        ) -> str:
            """Get weather forecast for given coordinates."""
            # This would call the actual MCP weather server
            # For now, return a placeholder
            return f"Weather forecast for coordinates ({latitude}, {longitude}) from {start_date} to {end_date}"

        @self.agent.tool
        async def get_locations(
            ctx: RunContext[WeatherAgentDeps],
            name: str,
            country_code: Optional[str] = None,
            count: int = 10,
        ) -> str:
            """Get location information for a place name."""
            # This would call the actual MCP weather server
            # For now, return a placeholder
            return f"Location search results for '{name}' in {country_code or 'any country'} (max {count} results)"

    async def chat(self, message: str) -> str:
        """Chat with the weather agent.

        Args:
            message: User message to send to the agent

        Returns:
            Agent's response
        """
        try:
            logger.info(f"Processing user message: {message}")

            # Run the agent with the user message
            result = await self.agent.run(message, deps=self.deps)

            logger.info("Agent response generated successfully")
            return str(result)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I apologize, but I encountered an error while processing your request: {e}"

    def cleanup(self) -> None:
        """Clean up MCP connections."""
        logger.info("Cleaning up MCP connections")
        # In a full implementation, we would close actual MCP connections here


# --------------------------------------------------------------------------------------
# CLI Interface
# --------------------------------------------------------------------------------------


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
def cli(log_level: str):
    """Weather Agent CLI."""
    # Update logging level if different from default
    if log_level.upper() != "INFO":
        level = getattr(logging, log_level.upper(), logging.INFO)
        colorized_config(level=level)


@cli.command()
@click.option(
    "--mcp-server-command",
    multiple=True,
    help="MCP server command (can be specified multiple times)",
)
@click.option("--interactive", "-i", is_flag=True, help="Run in interactive mode")
@click.argument("message", required=False)
def chat(mcp_server_command: tuple, interactive: bool, message: Optional[str]):
    """Chat with the weather agent."""

    # Parse MCP server configurations
    mcp_servers = []
    for i, command in enumerate(mcp_server_command):
        server_config = MCPServerConfig(
            name=f"server_{i}", command=command.split(), env=None
        )
        mcp_servers.append(server_config)

    if not mcp_servers:
        # Default MCP server configuration for weather server
        default_weather_server = MCPServerConfig(
            name="weather_server",
            command=[
                "python",
                "-m",
                "tools.mcp.weather_server.weather_server",
                "stdio",
            ],
            env=None,
        )
        mcp_servers.append(default_weather_server)

    # Create agent configuration
    config = WeatherAgentConfig(
        mcp_servers=mcp_servers,
        openai_api_key=None,  # Will be loaded from environment
        openai_api_base=None,  # Will be loaded from environment
    )

    # Initialize the weather agent
    try:
        agent = WeatherAgent(config)
        logger.info("Weather agent initialized successfully")

        if interactive:
            # Interactive mode
            click.echo("Weather Agent - Interactive Mode")
            click.echo("Type 'quit' or 'exit' to stop")
            click.echo("-" * 40)

            async def interactive_loop():
                while True:
                    try:
                        user_input = click.prompt("You", type=str)

                        if user_input.lower() in ["quit", "exit"]:
                            break

                        response = await agent.chat(user_input)
                        click.echo(f"Agent: {response}")
                        click.echo()

                    except (EOFError, KeyboardInterrupt):
                        break

            try:
                asyncio.run(interactive_loop())
            finally:
                agent.cleanup()

        elif message:
            # Single message mode
            async def single_message():
                response = await agent.chat(message)
                click.echo(response)

            try:
                asyncio.run(single_message())
            finally:
                agent.cleanup()

        else:
            click.echo("Please provide a message or use --interactive mode")

    except Exception as e:
        logger.error(f"Failed to initialize weather agent: {e}")
        raise click.ClickException(f"Failed to initialize weather agent: {e}") from e


@cli.command()
def test():
    """Test the weather agent with a sample query."""

    # Create test configuration
    test_mcp_server = MCPServerConfig(
        name="weather_server",
        command=["python", "-m", "tools.mcp.weather_server.weather_server", "stdio"],
        env=None,
    )
    config = WeatherAgentConfig(
        mcp_servers=[test_mcp_server],
        openai_api_key=None,  # Will be loaded from environment
        openai_api_base=None,  # Will be loaded from environment
    )

    try:
        agent = WeatherAgent(config)
        logger.info("Weather agent test initialized successfully")

        async def run_test():
            test_message = "What's the weather like in San Francisco, CA today?"
            click.echo(f"Test query: {test_message}")
            response = await agent.chat(test_message)
            click.echo(f"Response: {response}")

        try:
            asyncio.run(run_test())
        finally:
            agent.cleanup()

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise click.ClickException(f"Test failed: {e}") from e


if __name__ == "__main__":
    cli()
