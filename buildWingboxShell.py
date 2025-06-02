#
# File: buildWingboxShell.py
# Name: Omar Abdallah
# Class: AA 222 / CS 361
# Project: MESC Wing-Box Optimization
#

import json, sys, math
from abaqus import mdb, session
from abaqusConstants import *
import mesh, regionToolset

# ------------------------------------------------------------
# 0  Parameters (JSON override optional)
# ------------------------------------------------------------
par = dict(L=1.50, b=0.30, h=0.40, t_skin=3.0e-3, seed=0.10)
if len(sys.argv) > 2 and sys.argv[1] == '--':
    par.update(json.load(open(sys.argv[2])))

L, b, h   = par['L'], par['b'], par['h']
t_skin    = par['t_skin']
seed_size = par['seed']
job_name  = 'wingbox_shell'
EPS       = 1e-6                      # geometric tolerance

# ------------------------------------------------------------
# 1  Model & shell geometry
# ------------------------------------------------------------
model = mdb.Model(name='Wingbox')
sk    = model.ConstrainedSketch(name='xsec', sheetSize=1.0)
sk.rectangle((0., 0.), (b, h))
part  = model.Part(name='WB', dimensionality=THREE_D, type=DEFORMABLE_BODY)
part.BaseShellExtrude(sketch=sk, depth=L)

# ------------------------------------------------------------
# 2  Material & shell section (isotropic placeholder)
# ------------------------------------------------------------
mat = model.Material(name='CFRP_iso')
mat.Density(table=((1600.,),))
mat.Elastic(table=((135e9, 0.30),))
model.HomogeneousShellSection(name='Skin', material='CFRP_iso', thickness=t_skin)
faces_all = regionToolset.Region(faces=part.faces)
part.SectionAssignment(region=faces_all, sectionName='Skin')

# ------------------------------------------------------------
# 3  Assembly, boundary sets
# ------------------------------------------------------------
asm  = model.rootAssembly
inst = asm.Instance(name='WBinst', part=part, dependent=ON)

# Root edges: capture all edges whose centroid Z≈0
root_edges = inst.edges.getByBoundingBox(zMin=-EPS, zMax=EPS)
if not root_edges:
    raise ValueError('No root edges captured at z=0; check geometry.')
asm.Set(edges=root_edges, name='SET_FIXED')
model.EncastreBC(name='FixedRoot', createStepName='Initial', region=asm.sets['SET_FIXED'])

# Reference point at mid‑span, top skin
rp_id = asm.ReferencePoint((b/2., h, L/2.)).id
rp    = asm.referencePoints[rp_id]
asm.Set(referencePoints=(rp,), name='SET_RP')

# Top surface for coupling (single face wrapped as tuple)
top_face = inst.faces.findAt(((b/2., h, L/2.),))
asm.Surface(side1Faces=(top_face,), name='TopSurf')
model.Coupling(name='CoupTop', controlPoint=asm.sets['SET_RP'],
               surface=asm.surfaces['TopSurf'], couplingType=KINEMATIC,
               influenceRadius=WHOLE_SURFACE)

# ------------------------------------------------------------
# 4  Step & load
# ------------------------------------------------------------
model.StaticStep(name='BEND', previous='Initial')
M_unit = 1.0e4                       # 10 kN·m
model.Moment(name='UnitM', createStepName='BEND', region=asm.sets['SET_RP'], cm1=M_unit)

# ------------------------------------------------------------
# 5  Meshing
# ------------------------------------------------------------
part.seedPart(size=seed_size)
part.setElementType(regions=faces_all, elemTypes=(mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD),))
part.generateMesh()

# ------------------------------------------------------------
# ------------------------------------------------------------
# 6  Compute mass (pre‑solve) & Job
# ------------------------------------------------------------
# getMassProperties on the Assembly works before the job runs
mass = asm.getMassProperties()['mass']

job = mdb.Job(name=job_name, model='Wingbox')
job.submit(); job.waitForCompletion()

# ------------------------------------------------------------
job = mdb.Job(name=job_name, model='Wingbox')
job.submit(); job.waitForCompletion()

# ------------------------------------------------------------
# 7  Post‑processing
# ------------------------------------------------------------
from odbAccess import openOdb
odb   = openOdb(job_name + '.odb')
frame = odb.steps['BEND'].frames[-1]
uz = frame.fieldOutputs['U'].getSubset(
        region=odb.rootAssembly.nodeSets['SET_RP']).values[0].data[2]
EI = M_unit / uz
mass = odb.rootAssembly.getMassProperties()['mass']
odb.close()

print('--- RESULTS ----------------------------')
print(f't_skin (mm)  = {t_skin*1e3:.3f}')
print(f'U3 (m)       = {uz:.6e}')
print(f'EI (N·m^2)   = {EI:.3e}')
print(f'Mass (kg)    = {mass:.3f}')
print('----------------------------------------')
