import time
from datetime import datetime

import numpy as np
import random
import cv2  # Para carregar a imagem alvo
import matplotlib
matplotlib.use('Agg')  # Configura o backend para 'Agg'
import matplotlib.pyplot as plt

TIME_LIMIT = 60
POP_SIZE = 100
CROMOSSOME_LENGTH = 90
MUTATION_RATE = 0.1
MUTATION_RATE_DECAY = 0.95  # Fator de decaimento da mutação por geração
NUM_GENERATIONS = 200
ELITE_SIZE = 5  # Número de indivíduos mantidos por elitismo


def chromosome_to_image(chromosome):
    img = np.ones((64, 64), dtype=np.float32) * 255  # Cria uma imagem branca de 64x64 pixels
    num_circles = len(chromosome) // 3
    for i in range(num_circles):
        x = int(chromosome[3 * i] % 64)
        y = int(chromosome[3 * i + 1] % 64)
        r = int(abs(chromosome[3 * i + 2]) % 10)  # Raio máximo 10
        y_grid, x_grid = np.ogrid[:64, :64]
        mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= r ** 2
        img[mask] = 0  # Círculos pretos
    return img


def fitness_function(chromosome):
    generated_img = chromosome_to_image(chromosome)
    mse = np.mean((generated_img - TARGET_IMG) ** 2)
    return mse


# Função de seleção por torneio
def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1])  # Ordena pelo menor MSE
    return selected[0][0]


# Função de crossover
def crossover(parent1, parent2):
    point = random.randint(0, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2


# Função de mutação
def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] += np.random.normal(0, 1)  # Mutação gaussiana
    return chromosome


def genetic_algorithm(population, image_name, img_extension):
    print()
    print(f"Início: {datetime.now()}")
    start_time = time.time()
    current_mutation_rate = MUTATION_RATE  # Taxa de mutação inicial

    for generation in range(NUM_GENERATIONS):
        moment_time = time.time() - start_time
        if moment_time > TIME_LIMIT:
            print(datetime.now())
            return best_individual

        fitnesses = [fitness_function(ind) for ind in population]
        new_population = [ind for ind in sorted(population, key=fitness_function)[:ELITE_SIZE]]  # Elitismo

        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, MUTATION_RATE)
            child2 = mutate(child2, MUTATION_RATE)
            new_population.extend([child1, child2])

        population = new_population[:POP_SIZE]

        # Melhor solução da geração
        best_individual = sorted(population, key=fitness_function)[0]
        best_fitness = fitness_function(best_individual)
        print(f'Generation {generation}, Best MSE: {best_fitness}')

        if generation % 10 == 0:
            intermediate_image = chromosome_to_image(best_individual)
            drawImage(intermediate_image, f'{image_name}_generation_{generation}', img_extension)

        if best_fitness < 100:
            print(datetime.now())
            return best_individual

        # Atualiza a taxa de mutação com decaimento
        current_mutation_rate *= MUTATION_RATE_DECAY

    return best_individual

def create_population():
    return [np.random.rand(CROMOSSOME_LENGTH) * 64 for _ in range(POP_SIZE)]


def drawImage(best_image, image_name, image_extension):
    plt.imshow(best_image, cmap='gray', aspect='auto')
    plt.axis('off')  # Remove os eixos
    plt.gcf().set_size_inches(1, 1)  # Define o tamanho da figura como 1x1 polegada
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove espaços em branco
    plt.savefig(f'resources/out/{image_name}{image_extension}', format='jpg', dpi=64, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_all():
    images_names = ['c', 'coracao', 'estrela', 'raio', 'rosto', 'seta']

    for image in images_names:
        img_extension = '.jpg'
        TARGET_IMG = cv2.imread(f'resources/in/{image}{img_extension}', cv2.IMREAD_GRAYSCALE)  # Carregar imagem alvo
        population = create_population()
        best_individual = genetic_algorithm(population, image, img_extension)
        best_image = chromosome_to_image(best_individual)
        drawImage(best_image, image, img_extension)


image_name = 'estrela'
img_extension = '.jpg'
TARGET_IMG = cv2.imread(f'resources/in/{image_name}{img_extension}', cv2.IMREAD_GRAYSCALE)  # Carregar imagem alvo

population = create_population()
best_individual = genetic_algorithm(population, image_name, img_extension)
best_image = chromosome_to_image(best_individual)
drawImage(best_image, image_name, img_extension)

process_all()
