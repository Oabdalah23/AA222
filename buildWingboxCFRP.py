#
# File: buildWingboxShell.py
# Name: Omar Abdallah
# Class: AA 222 / CS 361
# Project: MESC Wing-Box Optimization
# Note: Attempted to Construct Abaqus High Fidelity Model
#

import sys, json, uuid
from abaqus import mdb
from abaqusConstants import *
import mesh, regionToolset

P = dict(L=1.50, b=0.30, h=0.40,           # [m]
         t_ply=0.125e-3,                   # [m]
         angles=[0, 45],                   # half-stack (deg)
         symmetric=True,
         seed=0.10)

if len(sys.argv) > 2 and sys.argv[1] == '--':
    P.update(json.load(open(sys.argv[2])))

L, b, h   = P['L'], P['b'], P['h']
t_ply     = P['t_ply']
stack     = P['angles'] + (P['angles'][::-1] if P['symmetric'] else [])
seed      = P['seed']
EPS       = 1e-6
job_name  = 'WingBoxCFRPRun'

if 'Wingbox' in mdb.models:  del mdb.models['Wingbox']
model = mdb.Model('Wingbox')

sk = model.ConstrainedSketch('xsec', 1.0)
sk.rectangle((0, 0), (b, h))
part = model.Part('WB', THREE_D, DEFORMABLE_BODY)
part.BaseShellExtrude(sketch=sk, depth=L)

mat = model.Material('AS4_8552')
mat.Density(((1570.,),))
mat.Elastic(type=ENGINEERING_CONSTANTS,
            table=((135e9, 10e9, 10e9, 0.30, 0.30, 0.25, 5.2e9, 5.2e9, 3.9e9),))

reg_all = regionToolset.Region(faces=part.faces)
lay = part.CompositeLayup('Layup', elementType=SHELL,
                          symmetric=False, thicknessAssignment=FROM_SECTION)
tag = uuid.uuid4().hex[:6]
for i, ang in enumerate(stack, 1):
    lay.CompositePly(region=reg_all, plyName=f'P{i}_{tag}',
                     material='AS4_8552',
                     thicknessType=SPECIFY_THICKNESS, thickness=t_ply,
                     orientationType=SPECIFY_ORIENT, orientationValue=float(ang),
                     numIntPoints=3)

a   = model.rootAssembly
inst = a.Instance('WBinst', part, dependent=ON)

root_edges = inst.edges.getByBoundingBox(zMin=-EPS, zMax=EPS)
a.Set(edges=root_edges, name='FIXED')
model.EncastreBC('BC_Root', 'Initial', a.sets['FIXED'])

# Reference point & node-set
rp_id   = a.ReferencePoint((b/2., h, L)).id
rp_node = a.referencePoints[rp_id]
a.Set(name='SET_RP', referencePoints=(rp_node,))   # true node set

# Tip surface
tip_face = inst.faces.findAt(((b/2., h-EPS, L-EPS),))
a.Surface(side1Faces=(tip_face,), name='TopSurf')

# Distributing coupling (structural) ★
a.model.Coupling(name='CPL_Tip',
                               controlPoint=a.sets['SET_RP'],
                               surface=a.surfaces['TopSurf'],
                               influenceRadius=WHOLE_SURFACE,
                               couplingType=STRUCTURAL,
                               u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)

model.StaticStep('BEND', 'Initial', nlgeom=OFF)
M_unit = 1.0e4                                   # 10 kN·m about +X

# Moment & history output applied to the **node set** ★★
rp_set = a.sets['SET_RP']
model.Moment('UnitM', 'BEND', region=rp_set, cm1=M_unit)
model.HistoryOutputRequest(name='H-RP', createStepName='BEND',
                           variables=('U3',), region=rp_set)

part.seedPart(seed)
elem = mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD)
part.setElementType(regions=reg_all, elemTypes=(elem,))
part.generateMesh()

if job_name in mdb.jobs: del mdb.jobs[job_name]
mdb.Job(job_name, model=model.name).submit().waitForCompletion()

from odbAccess import openOdb
odb = openOdb(job_name + '.odb')
rp   = odb.rootAssembly.nodeSets['SET_RP']
uz   = odb.steps['BEND'].frames[-1].fieldOutputs['U'] \
         .getSubset(region=rp).values[0].data[2]

if abs(uz) < 1e-15:
    odb.close()
    raise RuntimeError('Zero RP deflection – check coupling / load')

EI   = M_unit / uz
mass = odb.rootAssembly.getMassProperties()['mass']
odb.close()

print('--- RESULTS --------------------------------')
print(f'Stack   : {stack}  (deg, top→bottom)')
print(f'Ply t   : {t_ply*1e3:.3f} mm  ×  {len(stack)} plies')
print(f'U3 (m)  : {uz:.6e}')
print(f'EI (N·m²): {EI:.3e}')
print(f'Mass (kg): {mass:.3f}')
print('--------------------------------------------')