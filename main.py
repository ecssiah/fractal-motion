import numpy as np
import matplotlib.pyplot as plt

escape_radius_squared = 8.0
points = 1000000
iterations = 1000
dimensions = (800, 800)
x_limits = (-2.0, 2.0)
y_limits = (-2.0, 2.0)

I = np.array([
    [ 1,  0],
    [ 0,  1],
])

R = np.array([
    [ 0,  1],
    [ 1,  0],
])

B = np.array([
    [ 0,  1],
    [-1,  0],
])

G = np.array([
    [ 1,  0],
    [ 0, -1],
])

def generate():
    pixel_array = np.zeros(dimensions, dtype=int)

    for _ in range(points):
        path = []

        c_angle = np.random.uniform(0, 2 * np.pi)
        c_radius = np.random.uniform(0, x_limits[1])
        
        cx = c_radius * np.cos(c_angle)
        cy = c_radius * np.sin(c_angle)

        C = cx * I + cy * B

        z = C

        for _ in range(iterations):
            z = z @ z + C

            magnitude_squared = np.sum(np.square(z))

            if magnitude_squared < escape_radius_squared:
                path.append(z)
            else:
                for point in path:
                    cell_x = int((point[0, 0] - x_limits[0]) / (x_limits[1] - x_limits[0]) * (dimensions[0] - 1))
                    cell_y = int((point[0, 1] - y_limits[0]) / (y_limits[1] - y_limits[0]) * (dimensions[1] - 1))

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
