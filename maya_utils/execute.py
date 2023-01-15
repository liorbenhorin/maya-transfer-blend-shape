import maya.cmds as cmds
import pymel.core as pm

import transfer_blend_shape

source = pm.PyNode("body_templateBase")
bsp_node = pm.PyNode("Mocap_Blendshapes")
names = cmds.listAttr("Mocap_Blendshapes.w", m=True)
sources_grp = pm.group(em=True, n="sources")
sources = []
for i, n in enumerate(names):
    bsp_node.weight[i].set(1)
    new = next(iter(pm.duplicate(source)))
    new.rename(n)
    pm.parent(new, sources_grp)
    sources.append(new)
    bsp_node.weight[i].set(0)
    pm.select(cl=True)

source_mesh = "body_templateBase"
target_mesh = "Age71_M_H"
transfer = transfer_blend_shape.Transfer(source_mesh, target_mesh, virtual_mesh=None, iterations=3, threshold=0.001)

pm.select(cl=True)
results = []
for s in sources:
    n = s.name()
transfer.execute_from_mesh(s.name(), n)
results.append(pm.PyNode(n))

target_blendshape = pm.blendShape(*results, target_mesh)
