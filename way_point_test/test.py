import os
import matplotlib.pyplot as plt

folder_path = "./B1"

plt.figure(figsize=(8, 6))

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        x_coords = []
        y_coords = []

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4 and parts[1] == "TYPE_WAYPOINT":
                    try:
                        x = float(parts[2])
                        y = float(parts[3])
                        x_coords.append(x)
                        y_coords.append(y)
                    except ValueError:
                        continue

        if x_coords and y_coords:
            plt.plot(x_coords, y_coords, marker='.', label=filename)

plt.title("Trajectories from All Files")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()