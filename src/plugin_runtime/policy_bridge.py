"""Typed host-to-plugin policy requests, without importing plugin implementation code."""

from typing import Any

from src.plugin_runtime.host.hook_spec_registry import HookSpec, HookSpecRegistry


def register_policy_hook_specs(registry: HookSpecRegistry) -> list[HookSpec]:
    return registry.register_hook_specs(
        [
            HookSpec(
                name="extension.policy.resolve",
                description="Resolve an extension-owned policy or transformation.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "feature": {"type": "string"},
                        "action": {"type": "string"},
                        "payload": {"type": "object"},
                        "handled": {"type": "boolean"},
                        "result": {},
                    },
                    "required": ["feature", "action", "payload", "handled"],
                },
                default_timeout_ms=3000,
                allow_abort=True,
                allow_kwargs_mutation=True,
            )
        ]
    )


async def resolve_policy(feature: str, action: str, payload: dict[str, Any]) -> tuple[bool, Any]:
    from src.plugin_runtime.integration import get_plugin_runtime_manager

    result = await get_plugin_runtime_manager().invoke_hook(
        "extension.policy.resolve",
        feature=feature,
        action=action,
        payload=payload,
        handled=False,
        result=None,
    )
    if result.aborted:
        raise RuntimeError(f"Extension policy aborted: {feature}/{action}")
    return bool(result.kwargs.get("handled")), result.kwargs.get("result")


async def resolve_mapping_policy(feature: str, action: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """An unhandled policy is optional; a handled decision must be an object."""
    handled, result = await resolve_policy(feature, action, payload)
    if not handled:
        return None
    if not isinstance(result, dict):
        raise ValueError(f"Invalid extension policy result: {feature}/{action} must return an object")
    return result


async def resolve_binary_policy(feature: str, action: str, data: bytes, metadata: dict[str, Any]) -> tuple[bool, Any]:
    import asyncio
    import tempfile

    with tempfile.NamedTemporaryFile() as asset:
        await asyncio.to_thread(asset.write, data)
        asset.flush()
        return await resolve_policy(feature, action, {**metadata, "asset_path": asset.name})
