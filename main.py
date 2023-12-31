import numpy as np
import matplotlib.pyplot as plt

radius = 8.0
points = 1000000
iterations = 100
dimensions = (800, 800)
x_limits = (-2 * np.pi, 2 * np.pi)
y_limits = (-2 * np.pi, 2 * np.pi)

def generate():
    pixel_array = np.zeros(dimensions, dtype=int)

    for _ in range(points):
        c_angle = np.random.uniform(0, 2 * np.pi)
        c_radius = np.random.uniform(0, x_limits[1])
        
        cx = c_radius * np.cos(c_angle)
        cy = c_radius * np.sin(c_angle)

        zx = cx
        zy = cy

        path = []
        
        for _ in range(iterations):
            zx = zx * zx - zy * zy + cx
            zy = 2.0 * zx * zy + cy

            magnitude_squared = zx * zx + zy * zy

            if magnitude_squared < radius:
                path.append((zx, zy))
            else:
                for point in path:
                    cell_x = int((point[0] - x_limits[0]) / (x_limits[1] - x_limits[0]) * (dimensions[0] - 1))
                    cell_y = int((point[1] - y_limits[0]) / (y_limits[1] - y_limits[0]) * (dimensions[1] - 1))

                    if cell_x >= 0 and cell_x <= dimensions[0] - 1 and cell_y >= 0 and cell_y <= dimensions[1] - 1:
                        pixel_array[cell_x, cell_y] += 1

                break

    return pixel_array

pixel_array = generate()

max_value = np.max(pixel_array)
normalized_pixel_array = np.log1p(pixel_array) / np.log1p(max_value)

plt.imshow(
    normalized_pixel_array, 
    cmap='hot',
    extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]]
)

plt.title('Fractal Motion')
plt.savefig('images/fractal.png')

plt.show()
