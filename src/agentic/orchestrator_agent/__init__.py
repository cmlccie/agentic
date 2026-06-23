"""Orchestrator Agent — config-driven LangGraph supervisor over A2A agents.

A sibling to the simple-agent that delegates work across a configured list of
downstream Agent2Agent (A2A) servers. Built with LangChain's `create_agent`
(subagents-as-tools supervisor pattern) and exposes an OpenAI-compatible API
under `/v1/` and an A2A server under `/a2a/`, with the same Kubernetes-native
configuration and hot-reload behavior as the simple-agent.
"""
