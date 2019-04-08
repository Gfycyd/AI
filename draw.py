from skimage.draw import random_shapes
from scipy.spatial import distance
from PIL import Image
import numpy as np
from skimage.draw import circle
from PIL import Image
import time
import numpy.random as rd

output = "output.jpg"
number = 10

def vector_to_img(vector):
   my_matrix = vector.reshape((512, 512, 3))
   img = Image.fromarray(np.uint8(my_matrix))
   img.show()
def img_to_vector(file):
   img = Image.open(file)
   img = np.array(img)
   fitness_vector = img.reshape((img.shape[0] * img.shape[1] * img.shape[2]))
   return (fitness_vector)


def generate_random_shape(img, learning_rate):
   image = img.reshape((512, 512, 3))
   centr = rd.randint(low=learning_rate, high=512 - learning_rate, size=(2,))
   print(centr)
   radius = rd.randint(low=1, high=learning_rate)
   rr, cc = circle(centr[0], centr[1], radius)
   color = rd.randint(low=0, high=256, size=(1, 3))
   print(image.shape)
   image[rr, cc] = color
   return image.reshape((512 * 512 * 3))


def fitness(x):
   return distance.euclidean(x, img_to_vector(output))


def initial_population(population_size, learning_rate):
   empty_img = np.ones((512, 512, 3)) * 255
   population = generate_random_shape(empty_img, learning_rate).reshape((512 * 512 * 3))
   for i in range(population_size - 1):
       population = np.vstack((population, generate_random_shape(empty_img, learning_rate)))

   return population


def selection(population):
   print('selection')
   costs = np.apply_along_axis(fitness, 1, population) # but it works just for 5, not for more or for less
   return np.array([population[np.argsort(costs)[:5][0]], population[np.argsort(costs)[:5][1]],
                    population[np.argsort(costs)[:5][2]], population[np.argsort(costs)[:5][3]],
                    population[np.argsort(costs)[:5][4]]
                    ])
   #return np.array(sorted(population, key=lambda rows: fitness(rows)))[:number/2]


def crossover(best_examples):
   print('crossover')
   return np.vstack(([best_examples] * 2))


def mutation(population, learning_rate):
   for i in range(population.shape[0]):
       population[i] = generate_random_shape(population[i], learning_rate)
   return population



def generation_algo():
   learning_rate = 50
   population = initial_population(number, learning_rate)
   print(population.shape)
   t1 = time.time()
   for i in range(1, 10000):
       top = selection(population)
       children = crossover(top)
       noise_children = mutation(children, learning_rate)
       population = np.vstack((top, noise_children))
       print('cost')
       print(fitness(top[0]))
       if i == 100 or i == 9999 or i == 5000 or i ==3000 :
           vector_to_img(top[0])

       if learning_rate >= 20:
           learning_rate -= 1

       print('iteration {}'.format(i))
   t2 = time.time() - t1
   print("Time: ")
   print(t2)


if name == "main":
    generation_algo()