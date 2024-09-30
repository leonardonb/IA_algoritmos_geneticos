import numpy as np
import random
import cv2  # Para carregar a imagem alvo
import matplotlib
matplotlib.use('Agg')  # Configura o backend para 'Agg'
import matplotlib.pyplot as plt

# Parâmetros do Algoritmo Genético
POP_SIZE = 100  # Aumentar o tamanho da população
CROMOSSOME_LENGTH = 90  # Reduzir o número de posições no cromossomo
MUTATION_RATE = 0.05  # Aumentar a taxa de mutação inicial
MUTATION_RATE_DECAY = 0.99  # Taxa de decaimento da mutação
NUM_GENERATIONS = 100  # Aumentar o número de gerações
ELITE_SIZE = 5  # Manter número de indivíduos com elitismo
IMG = 'coracao.jpg'
TARGET_IMG = cv2.imread(f'resources/in/{IMG}', cv2.IMREAD_GRAYSCALE)  # Carregar imagem alvo
TARGET_IMG = cv2.resize(TARGET_IMG, (64, 64))  # Redimensionar para 64x64

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

# Função de mutação com decaimento adaptativo
def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] += np.random.normal(0, 1)  # Mutação gaussiana
    return chromosome

# Inicializar população
population = [np.random.rand(CROMOSSOME_LENGTH) * 64 for _ in range(POP_SIZE)]

# Evolução
for generation in range(NUM_GENERATIONS):
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

    # Ajuste adaptativo da taxa de mutação
    MUTATION_RATE *= MUTATION_RATE_DECAY

    if generation % 10 == 0:
        intermediate_image = chromosome_to_image(best_individual)
        plt.imshow(intermediate_image, cmap='gray')
        plt.savefig(f'resources/out/intermediate_generation_{generation}.jpg')

    if best_fitness < 100:
        break

# Exibir melhor solução
best_image = chromosome_to_image(best_individual)
plt.imshow(best_image, cmap='gray', aspect='auto')
plt.axis('off')  # Remove os eixos
plt.gcf().set_size_inches(1, 1)  # Define o tamanho da figura como 1x1 polegada
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove espaços em branco
plt.savefig(f'resources/out/{IMG}', format='jpg', dpi=64, bbox_inches='tight', pad_inches=0)
plt.close()  # Fecha a figura
