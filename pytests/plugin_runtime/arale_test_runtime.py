"""Offline SDK policy runner for migration integration tests (no network or QQ sends)."""
from pathlib import Path
from fnmatch import fnmatchcase
from pytests.plugin_runtime.test_arale_migration import plugin, IDS

class PolicyRuntime:
    def __init__(self,tmp_path):
        self.plugins={};self.routes=[]
        for ident in IDS:
            path=tmp_path/ident;path.mkdir(parents=True,exist_ok=True)
            p,_=plugin(ident,path);self.plugins[ident]=p
            for item in p.get_components():
                metadata=item.get('metadata',{})
                if metadata.get('hook')=='extension.policy.resolve':
                    for route in metadata.get('policy_routes',[]):
                        self.routes.append((route,p,getattr(p,metadata['handler_name'])))
        self.roots=list(self.plugins.values())
        from pytests.plugin_runtime.test_arale_migration import MODULES
        for ident,(owner,section) in MODULES.items():
            if owner not in self.plugins:
                # Network owner is configured but not started by offline policy tests.
                p,_=plugin(owner,tmp_path/owner)
                self.plugins[owner]=p
            p=self.plugins[owner]
            self.plugins[ident]=p.modules[section] if owner in {'Arale-policy','Arale-learning'} else getattr(p,section)
    async def start(self):
        for p in self.roots:await p.on_load()
    async def resolve(self,feature,action,payload):
        found=[fn for route,p,fn in self.routes if fnmatchcase(feature+'.'+action,route)]
        if len(found)>1:raise ValueError('Duplicate policy owner')
        if not found:return False,None
        result=await found[0](feature=feature,action=action,payload=payload)
        kwargs=result.get('modified_kwargs',{}) if isinstance(result,dict) else {}
        return bool(kwargs.get('handled')),kwargs.get('result')
