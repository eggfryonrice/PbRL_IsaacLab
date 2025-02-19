import csv
from pxr import Usd, UsdGeom, Gf

# Open the USD file
usd_file = "envs/walker2d/walker2d.usda"
# usd_file = "envs/humanoid/humanoid.usda"
stage = Usd.Stage.Open(usd_file)

# Prepare output data
output_data = []

# Iterate over all prims in the stage
for prim in stage.Traverse():
    if prim.GetTypeName() == "Capsule":
        name = prim.GetName()

        # Get radius and extent
        radius_attr = prim.GetAttribute("radius")
        extent_attr = prim.GetAttribute("extent")
        transform_attr = prim.GetAttribute("xformOp:transform")
        axis = prim.GetAttribute("axis").Get()

        radius = radius_attr.Get() if radius_attr.IsValid() else None
        extent = extent_attr.Get() if extent_attr.IsValid() else None
        transform = (
            transform_attr.Get() if transform_attr.IsValid() else Gf.Matrix4d(1.0)
        )

        # Consider only the x value of extent, set y and z to 0
        if axis == "X":
            start_point_raw = Gf.Vec3d(extent[0][0] + radius, 0, 0)
            end_point_raw = Gf.Vec3d(extent[1][0] - radius, 0, 0)
        if axis == "Y":
            start_point_raw = Gf.Vec3d(0, extent[0][1] + radius, 0)
            end_point_raw = Gf.Vec3d(0, extent[1][1] - radius, 0)
        if axis == "Z":
            start_point_raw = Gf.Vec3d(0, 0, extent[0][2] + radius)
            end_point_raw = Gf.Vec3d(0, 0, extent[1][2] - radius)

        # Apply transformation to extent points
        start_point = transform.Transform(start_point_raw)
        end_point = transform.Transform(end_point_raw)

        output_data.append(
            {
                "name": name,
                "radius": radius,
                "start_point": (start_point[0], start_point[1], start_point[2]),
                "end_point": (end_point[0], end_point[1], end_point[2]),
            }
        )
    if prim.GetTypeName() == "Sphere":
        name = prim.GetName()

        # Get radius
        radius_attr = prim.GetAttribute("radius")

        radius = radius_attr.Get() if radius_attr.IsValid() else None
        output_data.append(
            {
                "name": name,
                "radius": radius,
                "start_point": (0, 0, 0),
                "end_point": (0, 0, 0),
            }
        )

print(output_data)

# Write to CSV file
csv_file = "tmp/body.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.DictWriter(
        file, fieldnames=["name", "radius", "start_point", "end_point"]
    )
    writer.writeheader()
    for data in output_data:
        writer.writerow(data)
