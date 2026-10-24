from copy import deepcopy

async def extend_tool_spec(spec,context=None):
    from src.plugin_runtime.policy_bridge import resolve_policy
    handled,result=await resolve_policy("tool_spec",spec.name,{
        "definition":spec.to_llm_definition(),"official_quote_policy":True,
        "session_id":context.session_id if context else ""})
    if not handled:return spec
    definition=result.get("definition")
    if not isinstance(definition,dict) or definition.get("name")!=spec.name:raise ValueError("Invalid tool declaration extension")
    copy=deepcopy(spec)
    if "description" in definition:copy.description=str(definition["description"])
    if "parameters_schema" in definition:
        import jsonschema
        jsonschema.Draft202012Validator.check_schema(definition["parameters_schema"])
        copy.parameters_schema=deepcopy(definition["parameters_schema"])
    private=result.get("private_parameters",[])
    if not isinstance(private,list) or any(not isinstance(k,str) for k in private):raise ValueError("Invalid private extension parameters")
    copy.metadata["extension_private_parameters"]=private
    return copy
