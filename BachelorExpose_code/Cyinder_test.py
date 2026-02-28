import trimesh
import open3d as o3d



cyl1 = trimesh.primitives.Cylinder(radius=0.2, height=5.0, sections=64)
cyl2 = trimesh.primitives.Cylinder(radius=0.2, height=3.0, sections=64)

# build your mesh as before…
mesh = trimesh.util.concatenate([cyl1, cyl2])

# convert to Open3D and show
o3d_mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(mesh.vertices),
    o3d.utility.Vector3iVector(mesh.faces))
o3d_mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([o3d_mesh])
