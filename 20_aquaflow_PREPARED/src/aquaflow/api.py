from .treatment.chlorine_ct import compliant
from .network.isolation import isolated
from .meters.balance import balance
from .leak.localize import rank
from .workflow_service import WorkflowService
from .planning.source_blend_scenarios import BlendWorkbench
class AquaFlowService:
    def __init__(self):
        self.workflows=WorkflowService()
        self.blend_workbench=BlendWorkbench()
    def health(self):return {"service":"aquaflow","status":"ok","runtime_dependencies":0}
    def ct(self,rows,required):return compliant(rows,required)
    def isolate(self,nodes,pipes,closed,sources):return isolated(nodes,pipes,closed,sources)
    def water_balance(self,ins,outs,known_loss):return balance(ins,outs,known_loss)
    def leak_rank(self,rows):return rank(rows)
    def run_workflow(self,name,payload,received_at=None):return self.workflows.run(name,payload,received_at)
    def create_blend_scenario(self,name,sources,constraints,selection=None,unavailable=(),step=10.0,at=None):
        return self.blend_workbench.create_draft(name,sources,constraints,selection,unavailable,step,at)
    def get_blend_scenario(self,name):return self.blend_workbench.get(name)
    def submit_blend_scenario(self,name,at=None):return self.blend_workbench.submit(name,at)
    def approve_blend_scenario(self,name,at=None):return self.blend_workbench.approve(name,at)
    def reject_blend_scenario(self,name,at=None):return self.blend_workbench.reject(name,at)
    def recalculate_blend_scenario(self,name,unavailable=(),step=10.0,at=None):
        return self.blend_workbench.recalculate(name,unavailable,step,at)
