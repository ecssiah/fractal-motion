import time
import numpy as np
import matplotlib.pyplot as plt

ESCAPE_RADIUS_SQUARED = 2.0 ** 2
POINTS = 2 ** 22
ITERATIONS = 2 ** 10
FRAME_SIZE = 800
DOMAIN_SIZE = 4.0
DOMAIN_RADIUS = DOMAIN_SIZE / 2.0

I = np.array([
    [ 1,  0],
    [ 0,  1],
])

R = np.array([
    [ 0,  1],
    [ 1,  0],
])

G = np.array([
    [ 1,  0],
    [ 0, -1],
])

B = np.array([
    [ 0,  1],
    [-1,  0],
])

def conjugate(z):
    z[0, 1] *= -1
    z[1, 0] *= -1

    return z


def get_seed():
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(0, DOMAIN_RADIUS + np.finfo(float).eps)
    
    a = radius * np.cos(angle)
    b = radius * np.sin(angle)

    return (a, b)


def generate():
    pixel_array = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=int)

    for point_index in range(POINTS):
        if point_index % 20000 == 0:
            print(f"{int(point_index / POINTS * 100)}")

        path = []

        a, b = get_seed()

        C = a * I + b * B

        z = C

        for _ in range(ITERATIONS):
            z = np.linalg.matrix_power(conjugate(z), 2) + C

            magnitude_squared = np.sum(np.square(z[:, 0]))

            if magnitude_squared <= ESCAPE_RADIUS_SQUARED:
                path.append(z)
            else:
                for point in path:
                    x = point[0, 0]
                    y = point[0, 1]

                    cell_x = int((x + DOMAIN_RADIUS) / DOMAIN_SIZE * (FRAME_SIZE - 1))
                    cell_y = int((y + DOMAIN_RADIUS) / DOMAIN_SIZE * (FRAME_SIZE - 1))

                    in_x_bounds = cell_x >= 0 and cell_x < FRAME_SIZE
                    in_y_bounds = cell_y >= 0 and cell_y < FRAME_SIZE

                    if in_x_bounds and in_y_bounds:
                        symmetric_cell_y = int((-y + DOMAIN_RADIUS) / DOMAIN_SIZE * (FRAME_SIZE - 1))
                        
                        pixel_array[cell_x, cell_y] += 1
                        pixel_array[cell_x, symmetric_cell_y] += 1

                break

    return pixel_array

start_time = time.time()

pixel_array = generate()
normalized_pixel_array = np.log1p(pixel_array) / np.log1p(np.max(pixel_array))

end_time = time.time()

elapsed_time = end_time - start_time

print(f"Script execution time: {elapsed_time:.2f} seconds")

plt.imshow(
    normalized_pixel_array, 
    cmap='Purples_r',
    extent=[-DOMAIN_SIZE, DOMAIN_SIZE, -DOMAIN_SIZE, DOMAIN_SIZE]
)

plt.title('Fractal Motion')
plt.savefig('images/fractal.png')

plt.show()
