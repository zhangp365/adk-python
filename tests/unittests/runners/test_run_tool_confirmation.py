# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for HITL flows with different agent structures."""

import copy
from unittest import mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.flows.llm_flows.functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai.types import FunctionCall
from google.genai.types import FunctionResponse
from google.genai.types import GenerateContentResponse
from google.genai.types import Part
import pytest

from .. import testing_utils


def _create_llm_response_from_tools(
    tools: list[FunctionTool],
) -> GenerateContentResponse:
  """Creates a mock LLM response containing a function call."""
  parts = [
      Part(function_call=FunctionCall(name=tool.name, args={}))
      for tool in tools
  ]
  return testing_utils.LlmResponse(
      content=testing_utils.ModelContent(parts=parts)
  )


def _create_llm_response_from_text(text: str) -> GenerateContentResponse:
  """Creates a mock LLM response containing text."""
  return testing_utils.LlmResponse(
      content=testing_utils.ModelContent(parts=[Part(text=text)])
  )


def _test_request_confirmation_function(
    tool_context: ToolContext,
) -> dict[str, str]:
  """A test tool function that requests confirmation."""
  if not tool_context.tool_confirmation:
    tool_context.request_confirmation(hint="test hint for request_confirmation")
    return {"error": "test error for request_confirmation"}
  return {"result": f"confirmed={tool_context.tool_confirmation.confirmed}"}


def _test_request_confirmation_function_with_custom_schema(
    tool_context: ToolContext,
) -> dict[str, str]:
  """A test tool function that requests confirmation, but with a custom payload schema."""
  if not tool_context.tool_confirmation:
    tool_context.request_confirmation(
        hint="test hint for request_confirmation with custom payload schema",
        payload={
            "test_custom_payload": {
                "int_field": 0,
                "str_field": "",
                "bool_field": False,
            }
        },
    )
    return {"error": "test error for request_confirmation"}
  return {
      "result": f"confirmed={tool_context.tool_confirmation.confirmed}",
      "custom_payload": tool_context.tool_confirmation.payload,
  }


class BaseHITLTest:
  """Base class for HITL tests with common fixtures."""

  @pytest.fixture
  def runner(self, agent: BaseAgent) -> testing_utils.InMemoryRunner:
    """Provides an in-memory runner for the agent."""
    return testing_utils.InMemoryRunner(root_agent=agent)


class TestHITLConfirmationFlowWithSingleAgent(BaseHITLTest):
  """Tests the HITL confirmation flow with a single LlmAgent."""

  @pytest.fixture
  def tools(self) -> list[FunctionTool]:
    """Provides the tools for the agent."""
    return [FunctionTool(func=_test_request_confirmation_function)]

  @pytest.fixture
  def llm_responses(
      self, tools: list[FunctionTool]
  ) -> list[GenerateContentResponse]:
    """Provides mock LLM responses for the tests."""
    return [
        _create_llm_response_from_tools(tools),
        _create_llm_response_from_text("test llm response after tool call"),
        _create_llm_response_from_text(
            "test llm response after final tool call"
        ),
    ]

  @pytest.fixture
  def mock_model(
      self, llm_responses: list[GenerateContentResponse]
  ) -> testing_utils.MockModel:
    """Provides a mock model with predefined responses."""
    return testing_utils.MockModel(responses=llm_responses)

  @pytest.fixture
  def agent(
      self, mock_model: testing_utils.MockModel, tools: list[FunctionTool]
  ) -> LlmAgent:
    """Provides a single LlmAgent for the test."""
    return LlmAgent(name="root_agent", model=mock_model, tools=tools)

  @pytest.mark.asyncio
  @pytest.mark.parametrize("tool_call_confirmed", [True, False])
  async def test_confirmation_flow(
      self,
      runner: testing_utils.InMemoryRunner,
      agent: LlmAgent,
      tool_call_confirmed: bool,
  ):
    """Tests HITL flow where all tool calls are confirmed."""
    user_query = testing_utils.UserContent("test user query")
    events = await runner.run_async(user_query)
    tools = agent.tools

    expected_parts = [
        (
            agent.name,
            Part(function_call=FunctionCall(name=tools[0].name, args={})),
        ),
        (
            agent.name,
            Part(
                function_call=FunctionCall(
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    args={
                        "originalFunctionCall": {
                            "name": tools[0].name,
                            "id": mock.ANY,
                            "args": {},
                        },
                        "toolConfirmation": {
                            "hint": "test hint for request_confirmation",
                            "confirmed": False,
                        },
                    },
                )
            ),
        ),
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=tools[0].name,
                    response={"error": "test error for request_confirmation"},
                )
            ),
        ),
        (agent.name, "test llm response after tool call"),
    ]

    simplified = testing_utils.simplify_events(copy.deepcopy(events))
    for i, (agent_name, part) in enumerate(expected_parts):
      assert simplified[i][0] == agent_name
      assert simplified[i][1] == part

    ask_for_confirmation_function_call_id = (
        events[1].content.parts[0].function_call.id
    )
    user_confirmation = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=ask_for_confirmation_function_call_id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={"confirmed": tool_call_confirmed},
            )
        )
    )
    events = await runner.run_async(user_confirmation)

    expected_parts_final = [
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=tools[0].name,
                    response={"result": f"confirmed={tool_call_confirmed}"},
                )
            ),
        ),
        (agent.name, "test llm response after final tool call"),
    ]
    assert (
        testing_utils.simplify_events(copy.deepcopy(events))
        == expected_parts_final
    )


class TestHITLConfirmationFlowWithCustomPayloadSchema(BaseHITLTest):
  """Tests the HITL confirmation flow with a single agent, for custom confirmation payload schema."""

  @pytest.fixture
  def tools(self) -> list[FunctionTool]:
    """Provides the tools for the agent."""
    return [
        FunctionTool(
            func=_test_request_confirmation_function_with_custom_schema
        )
    ]

  @pytest.fixture
  def llm_responses(
      self, tools: list[FunctionTool]
  ) -> list[GenerateContentResponse]:
    """Provides mock LLM responses for the tests."""
    return [
        _create_llm_response_from_tools(tools),
        _create_llm_response_from_text("test llm response after tool call"),
        _create_llm_response_from_text(
            "test llm response after final tool call"
        ),
    ]

  @pytest.fixture
  def mock_model(
      self, llm_responses: list[GenerateContentResponse]
  ) -> testing_utils.MockModel:
    """Provides a mock model with predefined responses."""
    return testing_utils.MockModel(responses=llm_responses)

  @pytest.fixture
  def agent(
      self, mock_model: testing_utils.MockModel, tools: list[FunctionTool]
  ) -> LlmAgent:
    """Provides a single LlmAgent for the test."""
    return LlmAgent(name="root_agent", model=mock_model, tools=tools)

  @pytest.mark.asyncio
  @pytest.mark.parametrize("tool_call_confirmed", [True, False])
  async def test_confirmation_flow(
      self,
      runner: testing_utils.InMemoryRunner,
      agent: LlmAgent,
      tool_call_confirmed: bool,
  ):
    """Tests HITL flow with custom payload schema."""
    tools = agent.tools
    user_query = testing_utils.UserContent("test user query")
    events = await runner.run_async(user_query)

    expected_parts = [
        (
            agent.name,
            Part(function_call=FunctionCall(name=tools[0].name, args={})),
        ),
        (
            agent.name,
            Part(
                function_call=FunctionCall(
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    args={
                        "originalFunctionCall": {
                            "name": tools[0].name,
                            "id": mock.ANY,
                            "args": {},
                        },
                        "toolConfirmation": {
                            "hint": (
                                "test hint for request_confirmation with"
                                " custom payload schema"
                            ),
                            "confirmed": False,
                            "payload": {
                                "test_custom_payload": {
                                    "int_field": 0,
                                    "str_field": "",
                                    "bool_field": False,
                                }
                            },
                        },
                    },
                )
            ),
        ),
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=tools[0].name,
                    response={"error": "test error for request_confirmation"},
                )
            ),
        ),
        (agent.name, "test llm response after tool call"),
    ]

    simplified = testing_utils.simplify_events(copy.deepcopy(events))
    for i, (agent_name, part) in enumerate(expected_parts):
      assert simplified[i][0] == agent_name
      assert simplified[i][1] == part

    ask_for_confirmation_function_call_id = (
        events[1].content.parts[0].function_call.id
    )
    custom_payload = {
        "test_custom_payload": {
            "int_field": 123,
            "str_field": "test_str",
            "bool_field": True,
        }
    }
    user_confirmation = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=ask_for_confirmation_function_call_id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={
                    "confirmed": tool_call_confirmed,
                    "payload": custom_payload,
                },
            )
        )
    )
    events = await runner.run_async(user_confirmation)

    expected_response = {
        "result": f"confirmed={tool_call_confirmed}",
        "custom_payload": custom_payload,
    }
    expected_parts_final = [
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=tools[0].name,
                    response=expected_response,
                )
            ),
        ),
        (agent.name, "test llm response after final tool call"),
    ]
    assert (
        testing_utils.simplify_events(copy.deepcopy(events))
        == expected_parts_final
    )


class TestHITLConfirmationFlowWithResumableApp:
  """Tests the HITL confirmation flow with a resumable app."""

  @pytest.fixture
  def tools(self) -> list[FunctionTool]:
    """Provides the tools for the agent."""
    return [FunctionTool(func=_test_request_confirmation_function)]

  @pytest.fixture
  def llm_responses(
      self, tools: list[FunctionTool]
  ) -> list[GenerateContentResponse]:
    """Provides mock LLM responses for the tests."""
    return [
        _create_llm_response_from_tools(tools),
        _create_llm_response_from_text("test llm response after tool call"),
        _create_llm_response_from_text(
            "test llm response after final tool call"
        ),
    ]

  @pytest.fixture
  def mock_model(
      self, llm_responses: list[GenerateContentResponse]
  ) -> testing_utils.MockModel:
    """Provides a mock model with predefined responses."""
    return testing_utils.MockModel(responses=llm_responses)

  @pytest.fixture
  def agent(
      self, mock_model: testing_utils.MockModel, tools: list[FunctionTool]
  ) -> LlmAgent:
    """Provides a single LlmAgent for the test."""
    return LlmAgent(name="root_agent", model=mock_model, tools=tools)

  @pytest.fixture
  def runner(self, agent: LlmAgent) -> testing_utils.TestInMemoryRunner:
    """Provides an in-memory runner for the agent."""
    # Mark the app as resumable. So that the invocation will be paused after the
    # long running tool call.
    app = App(
        name="InMemoryRunner",  # Required for using TestInMemoryRunner.
        resumability_config=ResumabilityConfig(is_resumable=True),
        root_agent=agent,
    )
    return testing_utils.TestInMemoryRunner(app=app, app_name=None)

  @pytest.mark.asyncio
  async def test_pause_on_request_confirmation(
      self,
      runner: testing_utils.TestInMemoryRunner,
      agent: LlmAgent,
  ):
    """Tests HITL flow where all tool calls are confirmed."""
    events = await runner.run_async_with_new_session("test user query")

    # Verify that the invocation is paused after the long running tool call.
    # So that no intermediate function response and llm response is generated.
    assert testing_utils.simplify_events(copy.deepcopy(events)) == [
        (
            agent.name,
            Part(function_call=FunctionCall(name=agent.tools[0].name, args={})),
        ),
        (
            agent.name,
            Part(
                function_call=FunctionCall(
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    args={
                        "originalFunctionCall": {
                            "name": agent.tools[0].name,
                            "id": mock.ANY,
                            "args": {},
                        },
                        "toolConfirmation": {
                            "hint": "test hint for request_confirmation",
                            "confirmed": False,
                        },
                    },
                )
            ),
        ),
    ]
