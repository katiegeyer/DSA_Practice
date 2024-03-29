# import arcpy
# Workshop starter practice problems
x = 20
print(x)

############

x = 20 + 3
print(x)

############

x = 20
y = 3
z = x + y
print(x)
print(y)
print(z)

############ bug, data types (2.7 only)

x = 20
y = 3
print(x)
print(y)
print(x / y)

############

x = 20.0
y = 3.0
print(x)
print(y)
print(x / y)

############

x = 20.0
y = 3.0
z = 5.0
print(x * y - z)
print(x + y / z)
print(x // y)
print(x % y)
print(x ** y)

# 55
# 4.6... but really 20.6

############

h = 'Hello World'
print(h)

############ bug, no space

h = 'Hello'
hw = h + 'World'
print(hw)

############

h = 'Hello'
hw = h + ' World'
print(hw)

############

h = 'Hello'
h += ' World'
print(h)

############

a = True
b = False
print(a)
print(b)

############

a = True
b = False


# Set the workspace (change this to the path of your geodatabase)
# arcpy.env.workspace = "C:/path/to/your/geodatabase.gdb"

# List all feature classes in the geodatabase
# feature_classes = arcpy.ListFeatureClasses()

# Print the names of the feature classes
print("Feature Classes in the Geodatabase:")
for fc in feature_classes:
    print(fc)

import arcpy

# Set the workspace (change this to your local path)
arcpy.env.workspace = "C:/data/yourgeodatabase.gdb"

# List all feature classes in the geodatabase
feature_classes = arcpy.ListFeatureClasses()

# Loop through each feature class
for fc in feature_classes:
    print(f"Processing {fc}...")

    # Add a new field named 'NewField' of type 'DOUBLE'
    arcpy.AddField_management(fc, "NewField", "DOUBLE")

    # Calculate the new field values based on some condition or formula
    # Here, we'll simply copy values from an existing field named 'ExistingField'
    # You can replace this with your own logic
    arcpy.CalculateField_management(fc, "NewField", "!ExistingField!", "PYTHON3")

    print(f"Added and calculated 'NewField' in {fc}")

print("Processing completed.")
